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


def run_class(cell, cls_name, seeds, dev):
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
    for recipe in RECIPES:
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
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", default="esm2_35M", choices=sorted(MODELS))
    ap.add_argument("--class-index", type=int,
                    help="index into the sorted class list -- for a slurm array")
    ap.add_argument("--class-name")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--list-classes", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import torch
    d = panel()
    classes = sorted(set(d["pcls"].tolist()))
    if a.list_classes:
        for i, c in enumerate(classes):
            print(f"{i}\t{c}\t{(d['pcls'] == c).sum()}")
        return 0
    name = a.class_name or classes[a.class_index]
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    print(f"arm={a.arm} class={name} device={dev} seeds={a.seeds}", flush=True)
    res = run_class(a.arm, name, a.seeds, dev)
    res["device"] = dev
    res["seconds"] = round(time.time() - t0, 1)
    res["versions"] = {"torch": torch.__version__,
                       "transformers": __import__("transformers").__version__}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    f = OUTDIR / f"{a.arm}__{name}.json"
    f.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {f.relative_to(ROOT)} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
