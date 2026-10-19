#!/usr/bin/env python3
"""Study AY: does fine-tuning recover the classes the frozen probe cannot see?

`docs/FINETUNE_RECOVERY_PREREGISTRATION.md`, frozen before this file existed.

🔑 The abstract says "An AUROC of 0.974 is compatible with a screen that cannot see two whole
families" -- phage at 12.3% and beta-lactamase at 21.0%. 🔴 Every number in this project comes
from a FROZEN representation, so nobody has shown whether those failures are a fact about the
task or a fact about frozen probing.

🔒 `15e`'s `recover()` fold, unchanged. Only the SCORER changes: for the same seed the same
negatives split the same way, and the fine-tuned arm trains EsmForSequenceClassification on
the SEQUENCES of the trained positives plus the training negatives, thresholds on the same
held-out negatives, and reports the same flagged fraction. A paired comparison on one fold.

🔒 Study AP's lesson in reverse: AP found an unexamined C = 1.0 cost up to 24 points, so if
the fine-tune is given no equivalent care and loses, that is a bug of my own making. Learning
rate and epochs are selected by nested hold-out on the TRAINED classes only, over a fixed
grid, ONCE per held-out class -- the hyperparameters must not depend on the negative draw.
"""
import argparse
import json
import re
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results/v3"
OUTDIR = ROOT / "results/finetune_recovery"
MODELS = {"esm2_35M": "facebook/esm2_t12_35M_UR50D",
          "esm2_650M": "facebook/esm2_t33_650M_UR50D"}
# 🔒 Amendment 2: the mean_* recipes hold POOLING to the frozen arm's and move only the
# representation. The original three stay; `head_only` measures the CLS handicap.
RECIPES = ("head_only", "last2", "full", "mean_last2", "mean_full")
LR_GRID = (1e-5, 5e-5, 1e-4)
EPOCH_GRID = (3, 10)
SEEDS, FRAC, SPEC, MAX_LEN = 10, 0.40, 0.95, 1022
DEAD = ("beta_lactamase", "phage_peptidoglycan_hydrolase")
AY3_FPR = (0.04, 0.06)

sys.path.insert(0, str(ROOT / "src"))


def panel():
    man = json.load(open(V3 / "embedding_manifest_v3.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}

    def fa(p):
        d, k = {}, None
        for ln in open(p):
            if ln.startswith(">"):
                k = ln[1:].split()[0]
                d[k] = ""
            else:
                d[k] += ln.strip()
        return d

    Pseq = fa(ROOT / "data/sequences/toxins_positive_v3.fasta")
    Nseq = fa(ROOT / "data/sequences/benign_negatives_v3.fasta")
    pacc = [r["acc"] for r in man["positive_rows"]]
    nacc = [r["acc"] for r in man["negative_rows"]]
    # 🔒 row order is the embeddings' row order, so the frozen and fine-tuned arms see the
    # same proteins in the same positions. Going through the accession, never the position.
    return {
        "pcls": np.array([cls[a] for a in pacc]),
        "pseq": [Pseq[a] for a in pacc], "nseq": [Nseq[a] for a in nacc],
        "P": np.load(V3 / "embeddings_positive_v3.npy"),
        "N": np.load(V3 / "embeddings_negative_v3.npy"),
    }


def split_negs(n_neg, seed):
    """`15e`'s negative split, verbatim, so both arms hold out the same negatives."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_neg)
    cut = int(n_neg * FRAC)
    return perm[:cut], perm[cut:]          # held out (threshold), training


def _build(model_id, recipe, dev):
    """Return (model, forward_fn) for a recipe.

    🔴 The `mean_*` recipes cannot use `EsmForSequenceClassification`'s head slot, because
    `EsmForSequenceClassification.forward` calls `self.classifier(sequence_output)` and
    passes NO attention mask -- so a head doing `features.mean(dim=1)` would average over
    PADDING, making every score depend on which sequences shared its batch. 🔒 That is the
    same padding-dependence that is still unresolved in the ProtT5 gate, and it is not being
    reintroduced here. The mean recipes therefore run `EsmModel` directly and mask the mean.
    """
    import torch
    from torch import nn
    from transformers import EsmForSequenceClassification, EsmModel

    if not recipe.startswith("mean_"):
        mdl = EsmForSequenceClassification.from_pretrained(model_id, num_labels=2)

        def fwd(m, enc):
            return m(**enc).logits

        return mdl, fwd

    class MeanPoolClassifier(nn.Module):
        """🔒 The frozen arm's pooling, as a trainable model: a MASKED mean over token
        positions, then one hidden layer and a 2-way output."""

        def __init__(self, mid):
            super().__init__()
            self.esm = EsmModel.from_pretrained(mid, add_pooling_layer=False)
            h = self.esm.config.hidden_size
            self.dense = nn.Linear(h, h)
            self.drop = nn.Dropout(self.esm.config.hidden_dropout_prob)
            self.out = nn.Linear(h, 2)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            hs = self.esm(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            x = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            x = self.drop(x)
            x = torch.tanh(self.dense(x))
            return self.out(self.drop(x))

    mdl = MeanPoolClassifier(model_id)

    def fwd(m, enc):
        return m(**enc)

    return mdl, fwd


def finetune_scores(model_id, recipe, lr, epochs, train_seq, train_y, score_sets, dev):
    """Fit once, score every row set. Returns {name: probability vector}.

    🔒 One fit per (recipe, lr, epochs, seed); the three row sets -- held-out negatives,
    held-out class, and the inner validation split -- are scored from the same model.
    """
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl, fwd = _build(model_id, recipe, dev)
    last2_only = recipe in ("last2", "mean_last2")
    if recipe == "head_only" or last2_only:
        for p in mdl.esm.parameters():
            p.requires_grad = False
        if last2_only:
            for blk in mdl.esm.encoder.layer[-2:]:
                for p in blk.parameters():
                    p.requires_grad = True
    mdl = mdl.to(dev).train()
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=lr)
    prep = [" ".join(re.sub(r"[UZOB]", "X", s[:MAX_LEN])) for s in train_seq]
    B = 4
    for ep in range(epochs):
        idx = np.random.default_rng(1000 + ep).permutation(len(prep))
        for i in range(0, len(idx), B):
            b = idx[i:i + B]
            enc = tok([prep[j] for j in b], return_tensors="pt", padding=True).to(dev)
            lg = fwd(mdl, enc)
            loss = torch.nn.functional.cross_entropy(
                lg, torch.tensor([train_y[j] for j in b]).to(dev))
            loss.backward()
            opt.step()
            opt.zero_grad()
    mdl.eval()
    out = {}
    with torch.no_grad():
        for name, seqs in score_sets.items():
            ps = []
            for i in range(0, len(seqs), B):
                sl = [" ".join(re.sub(r"[UZOB]", "X", s[:MAX_LEN]))
                      for s in seqs[i:i + B]]
                enc = tok(sl, return_tensors="pt", padding=True).to(dev)
                lg = fwd(mdl, enc)
                ps.append(torch.softmax(lg, -1)[:, 1].float().cpu().numpy())
            out[name] = np.concatenate(ps) if ps else np.array([])
    del mdl
    return out


def _auroc(pos, neg):
    """Rank-based AUROC, ties averaged. 🔒 Used only to SELECT hyperparameters."""
    a = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average the ranks of tied values, or a model that outputs one constant scores 1.0
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return 0.0
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def select_hp(model_id, recipe, pseq, pcls, nseq, hi, tri, dev):
    """Nested LR/epoch selection on the TRAINED classes only, once per held-out class.

    🔒 The held-out class contributes nothing: the inner validation set is a fifth of the
    TRAINED positives, and the negatives come from the training half of seed 0's split.
    ⚠️ Chosen once per class, not per seed, because hyperparameters must not depend on the
    negative draw -- which is the frozen arm's own convention for `C`.
    """
    nte, ntr = split_negs(len(nseq), 0)
    rng = np.random.default_rng(0)
    pp = rng.permutation(len(tri))
    nval = max(1, len(tri) // 5)
    inner_val, inner_fit = tri[pp[:nval]], tri[pp[nval:]]
    best, best_score = (LR_GRID[1], EPOCH_GRID[0]), -1.0
    for lr in LR_GRID:
        for ep in EPOCH_GRID:
            tr_seq = [pseq[i] for i in inner_fit] + [nseq[i] for i in ntr]
            tr_y = [1] * len(inner_fit) + [0] * len(ntr)
            s = finetune_scores(model_id, recipe, lr, ep, tr_seq, tr_y,
                                {"val": [pseq[i] for i in inner_val],
                                 "neg": [nseq[i] for i in nte]}, dev)
            # 🔴 Amendment 1: this scored by recall at the 95th percentile, the same quantity
            # the study reports. On ~27 inner positives that recall moves in steps of 3.7pp
            # and TIES were the normal case -- `beta_lactamase` returned 0.630 for both
            # epoch settings -- so the selection resolved to whichever combination ran first.
            # 🔒 AUROC is a SELECTION criterion only; nothing reported changes, and it is far
            # more granular at this sample size.
            score = _auroc(s["val"], s["neg"])
            if score > best_score:
                best, best_score = (lr, ep), score
            print(f"      hp lr={lr:g} ep={ep} inner_auroc={score:.4f}", flush=True)
    return best, best_score


def run_class(cell, cls_name, seeds, dev, only_recipe=None):
    import torch
    d = panel()
    pcls, pseq, nseq = d["pcls"], d["pseq"], d["nseq"]
    hi = np.where(pcls == cls_name)[0]
    tri = np.setdiff1d(np.arange(len(pcls)), hi)
    recover = import_module("15e_sae_feature_space_lomo").recover
    model_id = MODELS[cell]
    out = {"arm": cell, "class": cls_name, "n_held_out": int(len(hi)),
           "n_trained_pos": int(len(tri)), "seeds": seeds, "recipes": {}}
    # 🔒 the frozen arm, reproduced in the same run rather than quoted
    frozen = [recover(d["P"], d["N"], hi, tri, 1.0, s, False) for s in range(seeds)]
    out["frozen"] = {"mean": float(np.mean(frozen)), "sd": float(np.std(frozen, ddof=1)),
                     "per_seed": [float(x) for x in frozen]}
    print(f"  {cls_name}: n={len(hi)}  frozen {np.mean(frozen)*100:.2f}%", flush=True)
    for recipe in ([only_recipe] if only_recipe else RECIPES):
        (lr, ep), inner = select_hp(model_id, recipe, pseq, pcls, nseq, hi, tri, dev)
        rr, ff = [], []
        for s in range(seeds):
            nte, ntr = split_negs(len(nseq), s)
            tr_seq = [pseq[i] for i in tri] + [nseq[i] for i in ntr]
            tr_y = [1] * len(tri) + [0] * len(ntr)
            sc = finetune_scores(model_id, recipe, lr, ep, tr_seq, tr_y,
                                 {"hi": [pseq[i] for i in hi],
                                  "neg": [nseq[i] for i in nte]}, dev)
            thr = float(np.quantile(sc["neg"], SPEC))
            rr.append(float((sc["hi"] >= thr).mean()))
            ff.append(float((sc["neg"] >= thr).mean()))
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        out["recipes"][recipe] = {
            "lr": lr, "epochs": ep, "inner_recall": inner,
            "mean": float(np.mean(rr)), "sd": float(np.std(rr, ddof=1)),
            "per_seed": [float(x) for x in rr],
            "realized_fpr": float(np.mean(ff)),
            # 🔒 AY-3: a recovery figure at an unmatched operating point is not comparable
            "AY3_fpr_in_band": bool(AY3_FPR[0] <= float(np.mean(ff)) <= AY3_FPR[1]),
            "delta_vs_frozen": float(np.mean(rr)) - out["frozen"]["mean"],
        }
        r = out["recipes"][recipe]
        print(f"    {recipe:10s} lr={lr:g} ep={ep}  {r['mean']*100:6.2f}%  "
              f"delta {r['delta_vs_frozen']*100:+6.2f}pp  fpr {r['realized_fpr']*100:.2f}%"
              f"{'' if r['AY3_fpr_in_band'] else '  AY-3 FAIL'}", flush=True)
    return out


def merge():
    """Assemble per-(class, recipe) files into one per class.

    🔒 The frozen arm is recomputed identically in every per-recipe run -- same fold, same
    seeds -- so the merge ASSERTS they agree rather than taking the first. A disagreement
    would mean the fold is not deterministic, which is the one thing this comparison rests on.
    """
    import collections
    by = collections.defaultdict(dict)
    frozen = {}
    for f in sorted(OUTDIR.glob("*__*__*.json")):
        d = json.loads(f.read_text())
        key = (d["arm"], d["class"])
        for r, v in d["recipes"].items():
            by[key][r] = v
        fm = d["frozen"]["mean"]
        if key in frozen and abs(frozen[key] - fm) > 1e-12:
            raise SystemExit(f"XX {key} frozen arm differs between per-recipe runs: "
                             f"{frozen[key]} vs {fm} -- the fold is not deterministic")
        frozen[key] = fm
        by[key]["__meta"] = {k: d[k] for k in
                             ("arm", "class", "n_held_out", "n_trained_pos", "seeds",
                              "frozen", "device", "versions") if k in d}
    n = 0
    for (arm, cls_name), recipes in by.items():
        meta = recipes.pop("__meta")
        out = dict(meta)
        out["recipes"] = recipes
        out["assembled_from_per_recipe_runs"] = sorted(recipes)
        (OUTDIR / f"{arm}__{cls_name}.json").write_text(json.dumps(out, indent=2) + "\n")
        n += 1
        print(f"  {arm}__{cls_name}: {len(recipes)} recipe(s)")
    print(f"assembled {n} class file(s)")
    return 0


AY1_BAND = (0.20, 0.07)                    # (artifact floor, survive ceiling)
AY2_BAND = (0.10, 0.03)


def band(delta, artifact_floor, survive_ceiling):
    """🔒 Two-sided, so BOTH readings are declarable.

    ⚠️ A band with only a floor can never fail and a band with only a ceiling can never
    pass. Study AW banded a difference at 1pp against intervals 4-9pp wide and its verdict
    became an artifact of that; these two came from the measured sd instead.
    """
    if delta >= artifact_floor:
        return "artifact"
    if delta <= survive_ceiling:
        return "survive"
    return "neither"


def load_merged():
    """The merged per-class files carry ONE `__`; the per-recipe files carry two."""
    arms = {}
    for f in sorted(OUTDIR.glob("*__*.json")):
        if f.stem.count("__") != 1:
            continue
        d = json.loads(f.read_text())
        arms.setdefault(d["arm"], {})[d["class"]] = d
    return arms


def matched_mean(cells, r1, r2):
    """🔴 `full` has 14 classes against `last2`'s 16, so their raw all-class means are not
    a comparison at all. Both are recomputed on the intersection, and the denominator is
    reported with them."""
    common = sorted(c for c, d in cells.items()
                    if r1 in d["recipes"] and r2 in d["recipes"]
                    and d["recipes"][r1]["AY3_fpr_in_band"]
                    and d["recipes"][r2]["AY3_fpr_in_band"])
    if not common:
        return None
    return {"n": len(common), "classes": common,
            r1: float(np.mean([cells[c]["recipes"][r1]["delta_vs_frozen"] for c in common])),
            r2: float(np.mean([cells[c]["recipes"][r2]["delta_vs_frozen"] for c in common]))}


def score_arm(cells):
    """AY-1, AY-2 and the AY-3 precondition, per recipe, for one arm."""
    recipes = sorted({r for d in cells.values() for r in d["recipes"]})
    out = {"n_classes": len(cells), "recipes": {}}
    for r in recipes:
        have = [c for c, d in cells.items() if r in d["recipes"]]
        # 🔒 AY-3 is a PRECONDITION, not a result: a recovery figure at an unmatched
        # operating point is not comparable, so a failing cell LEAVES the comparison and
        # the reduced denominator is stated rather than silently absorbed.
        bad = sorted(c for c in have if not cells[c]["recipes"][r]["AY3_fpr_in_band"])
        use = [c for c in have if c not in bad]

        def dl(c):
            return cells[c]["recipes"][r]["delta_vs_frozen"]

        dead = [c for c in DEAD if c in use]
        ay1 = float(np.mean([dl(c) for c in dead])) if len(dead) == len(DEAD) else None
        ay2 = float(np.mean([dl(c) for c in use])) if use else None
        out["recipes"][r] = {
            "AY1": ay1,
            "AY1_band": band(ay1, *AY1_BAND) if ay1 is not None else "unavailable",
            "AY2": ay2,
            "AY2_band": band(ay2, *AY2_BAND) if ay2 is not None else "unavailable",
            "AY2_n": len(use),
            "AY3_failed": bad,
            "classes_missing": sorted(set(cells) - set(have)),
            # 🔑 the mean of two classes hides its own mechanism -- one class can be
            # destroyed while the other triples -- so the parts go beside the mean, always.
            "per_dead_class": {c: {"frozen": cells[c]["frozen"]["mean"],
                                   "finetuned": cells[c]["recipes"][r]["mean"],
                                   "delta": dl(c)} for c in dead},
            "dead_spread": (abs(dl(dead[0]) - dl(dead[1]))
                            if len(dead) == len(DEAD) else None),
        }
    return out


def score():
    """Score AY-1..AY-3 and P1-P4 against the bands frozen on 2026-10-08."""
    arms = load_merged()
    if not arms:
        raise SystemExit("XX no merged class files -- run --merge first")
    out = {"bands": {"AY1": list(AY1_BAND), "AY2": list(AY2_BAND)},
           "arms": {a: score_arm(c) for a, c in sorted(arms.items())},
           # ⚠️ The preregistration states the bands once and does NOT name which recipe
           # carries them; amendment 2 then added the mean_* recipes. Reported per recipe
           # with the ambiguity stated, never resolved by picking one after the fact.
           "bands_name_no_recipe": True}

    # 🔒 Both arms must agree. `src/143_band_reachability.py` flags that two-unit
    # conjunction as a single-unit veto, accepted for study AW's reason: the second-arm
    # rule exists precisely so that one arm can veto.
    absent = sorted(set(MODELS) - set(arms))
    if absent:
        out["verdict"] = "undeclarable"
        out["verdict_reason"] = ("the preregistration requires both arms to agree; "
                                 f"{', '.join(absent)} has not run")
    else:
        labels = {}
        for r in sorted({r for a in out["arms"].values() for r in a["recipes"]}):
            labels[r] = {a: out["arms"][a]["recipes"].get(r, {}).get("AY1_band")
                         for a in out["arms"]}
        out["per_recipe_arm_labels"] = labels
        if any(len(set(v.values())) != 1 or None in v.values() for v in labels.values()):
            out["verdict"] = "arms disagree"
            out["verdict_reason"] = "one arm vetoes: the AY-1 band label differs by arm"
        else:
            seen = {next(iter(v.values())) for v in labels.values()}
            out["verdict"] = seen.pop() if len(seen) == 1 else "recipes disagree"

    out["predictions"] = score_predictions(out, arms, provisional=bool(absent))
    (OUTDIR / "AY_SCORED.json").write_text(json.dumps(out, indent=2) + "\n")
    report(out)
    return 0


def score_predictions(out, arms, provisional):
    """⚠️ Marked provisional while an arm is missing: the study verdict is undeclarable,
    so these are the numbers the present arm gives, not the study's scored record."""
    p = {"provisional": provisional}
    for arm, cells in sorted(arms.items()):
        rec = out["arms"][arm]["recipes"]
        got = {}
        avail = [v["AY1_band"] for v in rec.values() if v["AY1"] is not None]
        got["P1"] = {"kind": "reasoned",
                     "claim": 'AY-1 lands in "they survive", <= +7pp',
                     "holds": bool(avail) and all(b == "survive" for b in avail),
                     "per_recipe": {r: v["AY1_band"] for r, v in rec.items()}}
        m = matched_mean(cells, "full", "last2")
        got["P2"] = {"kind": "derived",
                     "claim": "`full` underperforms `last2` on the all-class mean",
                     "holds": None if m is None else bool(m["full"] < m["last2"]),
                     "matched": m}
        h = rec.get("head_only", {}).get("AY2")
        got["P3"] = {"kind": "derived",
                     "claim": "`head_only`'s all-class mean lands within 10pp of frozen",
                     "holds": None if h is None else bool(abs(h) <= 0.10),
                     "AY2_head_only": h}
        pos = [v["AY2"] for v in rec.values() if v["AY2"] is not None]
        got["P4"] = {"kind": "derived", "claim": "AY-2 is positive",
                     "holds": bool(pos) and all(x > 0 for x in pos),
                     "per_recipe": {r: v["AY2"] for r, v in rec.items()}}
        p[arm] = got
    return p


def report(out):
    print(f"verdict: {out['verdict']}"
          + (f"  ({out['verdict_reason']})" if "verdict_reason" in out else ""))
    for arm, a in out["arms"].items():
        print(f"\n{arm}  ({a['n_classes']} classes)")
        print(f"  {'recipe':<12}{'AY-1':>10} {'band':<12}{'AY-2':>10} {'band':<12}{'n':>3}"
              f"  {'spread':>8}")
        for r, v in a["recipes"].items():
            a1 = "   n/a" if v["AY1"] is None else f"{100*v['AY1']:+9.2f}pp"
            a2 = "   n/a" if v["AY2"] is None else f"{100*v['AY2']:+9.2f}pp"
            sp = "     n/a" if v["dead_spread"] is None else f"{100*v['dead_spread']:7.1f}pp"
            print(f"  {r:<12}{a1:>10} {v['AY1_band']:<12}{a2:>10} {v['AY2_band']:<12}"
                  f"{v['AY2_n']:>3}  {sp:>8}")
            if v["AY3_failed"]:
                print(f"    AY-3 left the comparison: {', '.join(v['AY3_failed'])}")
            if v["classes_missing"]:
                print(f"    never ran: {', '.join(v['classes_missing'])}")
    pr = out["predictions"]
    print(f"\npredictions{'  (PROVISIONAL -- an arm is missing)' if pr['provisional'] else ''}")
    for arm in [k for k in pr if k != "provisional"]:
        for name, v in pr[arm].items():
            mark = {True: "holds", False: "FAILS", None: "unavailable"}[v["holds"]]
            print(f"  {arm}  {name} ({v['kind']:<8}) {mark:<11} {v['claim']}")


def selftest():
    ok = True
    d = panel()
    if len(d["pseq"]) != 149 or len(d["nseq"]) != 296:
        print(f"  XX panel is {len(d['pseq'])}/{len(d['nseq'])}, expected 149/296")
        ok = False
    if d["P"].shape[0] != len(d["pseq"]) or d["N"].shape[0] != len(d["nseq"]):
        print("  XX embeddings and sequences disagree on row count")
        ok = False
    # 🔒 the negative split must be 15e's, so both arms hold out the SAME negatives
    m15 = import_module("15e_sae_feature_space_lomo")
    for s in (0, 7):
        nte, ntr = split_negs(296, s)
        rng = np.random.default_rng(s)
        perm = rng.permutation(296)
        cut = int(296 * m15.FRAC)
        if not (np.array_equal(nte, perm[:cut]) and np.array_equal(ntr, perm[cut:])):
            print(f"  XX the negative split at seed {s} is not 15e's")
            ok = False
    if (FRAC, SPEC) != (m15.FRAC, m15.SPEC):
        print(f"  XX FRAC/SPEC are {(FRAC, SPEC)}, 15e's are {(m15.FRAC, m15.SPEC)}")
        ok = False
    # 🔒 the frozen arm must reproduce the published numbers, or the fold is not the fold
    recover = m15.recover
    for c, want in (("beta_lactamase", 0.2119), ("phage_peptidoglycan_hydrolase", 0.1219)):
        hi = np.where(d["pcls"] == c)[0]
        tri = np.setdiff1d(np.arange(len(d["pcls"])), hi)
        got = float(np.mean([recover(d["P"], d["N"], hi, tri, 1.0, s, False)
                             for s in range(30)]))
        if abs(got - want) > 0.0005:
            print(f"  XX frozen {c} reproduces {got:.4f}, published {want}")
            ok = False
    # 🔒 the selection criterion must discriminate where recall ties, and a constant-output
    # model must score 0.5 rather than 1.0
    if abs(_auroc([3.0, 4.0], [1.0, 2.0]) - 1.0) > 1e-12:
        print(f"  XX a perfectly separating score is not AUROC 1.0: {_auroc([3,4],[1,2])}")
        ok = False
    if abs(_auroc([1.0, 2.0], [3.0, 4.0])) > 1e-12:
        print("  XX a perfectly inverted score is not AUROC 0.0")
        ok = False
    if abs(_auroc([1.0, 1.0], [1.0, 1.0]) - 0.5) > 1e-12:
        print(f"  XX a constant output is not AUROC 0.5: {_auroc([1,1],[1,1])}")
        ok = False
    if abs(_auroc([2.0, 3.0], [1.0, 4.0]) - 0.5) > 1e-12:
        print(f"  XX an interleaved score is not AUROC 0.5: {_auroc([2,3],[1,4])}")
        ok = False
    if len(DEAD) != 2 or not all(c in set(d["pcls"]) for c in DEAD):
        print(f"  XX the two dead classes are not in the panel: {DEAD}")
        ok = False
    # 🔒 Null calibration on the bands: each edge must FIRE, and the gap between the two
    # edges must be non-empty -- otherwise one reading is unreachable and the band is the
    # AW failure again. A check that cannot fail is not a check.
    for nm, (af, sc) in (("AY1", AY1_BAND), ("AY2", AY2_BAND)):
        if af <= sc:
            print(f"  XX {nm}'s artifact floor {af} is not above its survive ceiling {sc}")
            ok = False
        for val, want in ((af, "artifact"), (af + 0.01, "artifact"), (sc, "survive"),
                          (sc - 0.01, "survive"), ((af + sc) / 2, "neither")):
            if band(val, af, sc) != want:
                print(f"  XX {nm} band({val}) is {band(val, af, sc)}, expected {want}")
                ok = False
    # 🔒 and the verdict must refuse to declare on one arm, since the prereg requires both
    if len(MODELS) < 2:
        print(f"  XX MODELS holds {len(MODELS)} arm(s); the second-arm rule needs two")
        ok = False
    # 🔴 `full` ran on 14 classes against `last2`'s 16, so P2 must compare on the
    # intersection. A synthetic pair where the counts differ must come back matched.
    fake = {f"c{i}": {"frozen": {"mean": 0.0},
                      "recipes": {"last2": {"delta_vs_frozen": 0.10, "mean": 0.10,
                                            "AY3_fpr_in_band": True}}}
            for i in range(4)}
    for i in range(2):
        fake[f"c{i}"]["recipes"]["full"] = {"delta_vs_frozen": 0.02, "mean": 0.02,
                                           "AY3_fpr_in_band": True}
    m = matched_mean(fake, "full", "last2")
    if m is None or m["n"] != 2 or abs(m["last2"] - 0.10) > 1e-12:
        print(f"  XX matched_mean did not restrict to the intersection: {m}")
        ok = False
    # and a cell that failed AY-3 must leave the comparison
    fake["c0"]["recipes"]["full"]["AY3_fpr_in_band"] = False
    m2 = matched_mean(fake, "full", "last2")
    if m2 is None or m2["n"] != 1:
        print(f"  XX an AY-3 failure did not leave the matched comparison: {m2}")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="assemble per-(class, recipe) files into per-class ones")
    ap.add_argument("--score", action="store_true",
                    help="score AY-1..AY-3 and P1-P4 against the bands frozen 2026-10-08. "
                         "🔒 Refuses to declare a verdict while an arm is missing, because "
                         "the preregistration requires both arms to agree.")
    ap.add_argument("--arm", default="esm2_35M", choices=sorted(MODELS))
    ap.add_argument("--class-index", type=int,
                    help="index into the sorted class list -- for a slurm array")
    ap.add_argument("--class-name")
    ap.add_argument("--task-index", type=int,
                    help="index into the (class x recipe) product, for a slurm array")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--recipe", choices=RECIPES,
                    help="run ONE recipe, so a slurm array can be (class x recipe). "
                         "🔴 Three tasks of the per-class array timed out at 16h -- 80 "
                         "fine-tunes is 11-16h on 32 CPUs -- and one of them was "
                         "beta_lactamase, the class the paper's headline turns on. Sixteen "
                         "fine-tunes per task is ~3h and isolates failures.")
    ap.add_argument("--list-classes", action="store_true")
    ap.add_argument("--list-tasks", action="store_true",
                    help="print the (class, recipe) task list for an array")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.merge:
        return merge()
    if a.score:
        return score()

    import torch
    d = panel()
    classes = sorted(set(d["pcls"].tolist()))
    if a.list_classes:
        for i, c in enumerate(classes):
            print(f"{i}\t{c}\t{(d['pcls'] == c).sum()}")
        return 0
    if a.list_tasks:
        for i, (c, r) in enumerate([(c, r) for c in classes for r in RECIPES]):
            print(f"{i}\t{c}\t{r}")
        return 0
    if a.task_index is not None:
        pairs = [(c, r) for c in classes for r in RECIPES]
        if not 0 <= a.task_index < len(pairs):
            raise SystemExit(f"--task-index {a.task_index} outside 0..{len(pairs) - 1}")
        a.class_name, a.recipe = pairs[a.task_index]
    name = a.class_name or classes[a.class_index]
    # 🔒 Idempotent: a task whose output already exists exits 0 without refitting, so the
    # array can be resubmitted at a different CPU count without redoing finished work.
    # ⚠️ Measured, after two mis-sizings and a wrong hypothesis: ~10-11 min per fine-tune at
    # 16 and at 32 CPUs, ~20 min at 8. Flat from 16 to 32, so 32 spends 4x the cpu=300 cap
    # for nothing; 16 gives 18 concurrent tasks instead of 9.
    done = OUTDIR / (f"{a.arm}__{name}__{a.recipe}.json" if a.recipe
                     else f"{a.arm}__{name}.json")
    if done.exists():
        print(f"{done.name} already exists; nothing to do")
        return 0
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    print(f"arm={a.arm} class={name} device={dev} seeds={a.seeds}", flush=True)
    res = run_class(a.arm, name, a.seeds, dev, a.recipe)
    res["device"] = dev
    res["seconds"] = round(time.time() - t0, 1)
    res["versions"] = {"torch": torch.__version__,
                       "transformers": __import__("transformers").__version__}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    f = OUTDIR / (f"{a.arm}__{name}__{a.recipe}.json" if a.recipe
                  else f"{a.arm}__{name}.json")
    f.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {f.relative_to(ROOT)} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
