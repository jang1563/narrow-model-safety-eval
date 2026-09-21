#!/usr/bin/env python3
"""
50_reference_set_poisoning.py - can a benign reference set suppress ONE hazard class while every
                                aggregate an operator watches gets better?

Where this came from
--------------------
`src/37` asked whether adding near confusors to the negative set drives a failing class down. The
answer on the canonical arm was that it does not drive the FAILING classes down much, because they
are already inside the benign density, but it flattens the RECOVERED ones: `rip_rrna_glycosidase`
went 94.8% -> 51.9% when 500 pool proteins chosen for similarity to it were appended as negatives.

That is a security question rather than a modelling one, so it gets its own measurement. The
negative set is the input a screening operator is most likely to accept from outside, and nothing
in this construction is forged: every added protein is a genuine reviewed Swiss-Prot entry that
passes the panel's own hazard exclusions.

What the operator would see
---------------------------
The question is not only "does the target class drop" but "would anyone notice". So three
quantities are reported for each negative set, all out of sample:

    target recovery        LOMO on the targeted class, the thing being suppressed
    collateral recovery    LOMO on every OTHER class, each held out in its own turn
    pool false positives   the realized rate on all 8,259 benign pool proteins

⚠️ The collateral number must come from LOMO, not from scoring the training positives. An earlier
version of this measurement scored `P[tri]`, which is in sample and reports 100.0% whatever the
negative set does. That number is meaningless and was removed rather than published. Each class
here is held out in its own fit, exactly as `03b` does it.

PREREGISTERED, written before the sweep over classes
----------------------------------------------------
    P1  the targeted class drops by more than any other class does under the same negative set.
    P2  the pool false-positive rate does not rise relative to the random-supplement control, so
        the aggregate gives the operator no signal that anything happened.

    SUPPORTED     both hold. Targeted suppression is invisible to aggregate monitoring and the
                  per-class table is the only place it shows.
    REFUTED       either the collateral damage is comparable to the targeted damage, in which case
                  this is indiscriminate degradation rather than a targeted attack, or the false
                  positive rate moves enough to be caught by ordinary monitoring.

Usage:
    python src/50_reference_set_poisoning.py --arm esm2_650M --target rip_rrna_glycosidase
    python src/50_reference_set_poisoning.py --arm esm2_650M --all-targets
"""
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
sys.path.insert(0, str(ROOT / "src"))
m37 = import_module("37_negative_supplement_from_pool")

FRAC, SPEC, SEEDS, K = 0.40, 0.95, 30, 500
CONTAMINANT = "Q8X739"
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def lomo_all(P, pcls, Nset, classes, seeds=SEEDS):
    """Hold each class out in turn against a fixed negative set. Returns {class: mean recovery}."""
    out = {}
    for c in classes:
        hi = np.where(pcls == c)[0]
        tri = np.array([i for i in range(len(P)) if i not in set(hi.tolist())])
        out[c] = float(m37.recover(P, Nset, hi, tri, seeds=seeds).mean())
    return out


def pool_fp(P, pcls, Nset, POOL, target, seeds=SEEDS):
    """Realized false-positive rate on the whole pool, at the threshold the operator would set."""
    hi = np.where(pcls == target)[0]
    tri = np.array([i for i in range(len(P)) if i not in set(hi.tolist())])
    vals = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Nset))
        cut = int(len(Nset) * FRAC)
        nte, ntr = perm[:cut], perm[cut:]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        m.fit(np.vstack([P[tri], Nset[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
        thr = np.quantile(m.predict_proba(Nset[nte])[:, 1], SPEC)
        vals.append(float((m.predict_proba(POOL)[:, 1] >= thr).mean()))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=sorted(ARMS))
    ap.add_argument("--target", default="rip_rrna_glycosidase")
    ap.add_argument("--all-targets", action="store_true")
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    suf = ARMS[a.arm]

    man = json.load(open(V3 / f"embedding_manifest_v3{suf}.json"))
    pman = json.load(open(V3 / f"embedding_manifest_pool_large_{a.arm}.json"))
    assert man["model"] == pman["model"], f"{man['model']} != {pman['model']}"
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    P = np.load(V3 / f"embeddings_positive_v3{suf}.npy")
    Npan = np.load(V3 / f"embeddings_negative_v3{suf}.npy")
    POOL = np.load(V3 / f"embeddings_pool_large_{a.arm}.npy")

    acc = [r.split("|")[1] if "|" in r else r for r in pman["rows"]]
    keep = np.array([i for i, x in enumerate(acc) if x != CONTAMINANT])
    assert len(keep) == len(acc) - 1, f"{CONTAMINANT} not found in the pool"
    POOL = POOL[keep]

    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    classes = sorted({c for c in pcls if (pcls == c).sum() >= 3})
    targets = classes if a.all_targets else [a.target]
    for t in targets:
        assert t in classes, f"{t} not a class with at least three members"

    print(f"arm {a.arm} ({man['model']})  positives {len(P)}  panel negatives {len(Npan)}  "
          f"pool {len(POOL)} ({CONTAMINANT} dropped)")
    print(f"K={a.k} added negatives, {SEEDS} seeds, threshold at the {SPEC:.0%} quantile\n")

    base = lomo_all(P, pcls, Npan, classes)
    rng0 = np.random.default_rng(0)
    rnd = rng0.choice(len(POOL), a.k, replace=False)
    Nrnd = np.vstack([Npan, POOL[rnd]])
    base_rnd = lomo_all(P, pcls, Nrnd, classes)

    res = {}
    for t in targets:
        hi = np.where(pcls == t)[0]
        oi = np.where(pcls != t)[0]
        sim = m37.cos(POOL, P[hi]).max(axis=1)
        near = np.argsort(-sim)[:a.k]
        # "Nearest to t" is mostly "nearest to the panel", so it is not a targeted attack at all:
        # the per-class nearest-500 sets overlap at mean Jaccard 0.324 and reach 0.883 for
        # adp_ribosyl_ab_toxin against rip_rrna_glycosidase, and all thirteen together span only
        # 1,999 distinct pool proteins out of 6,500 slots. The selective version subtracts the
        # neighbourhood of every OTHER class, so it picks proteins near t and far from the rest.
        # Whether a genuinely targeted attack exists is decided by this mode, not by `nearest`.
        excl = np.argsort(-(sim - m37.cos(POOL, P[oi]).max(axis=1)))[:a.k]
        Nnear = np.vstack([Npan, POOL[near]])
        Nexcl = np.vstack([Npan, POOL[excl]])
        poisoned = lomo_all(P, pcls, Nnear, classes)
        selective = lomo_all(P, pcls, Nexcl, classes)
        jac = len(set(near.tolist()) & set(excl.tolist())) / len(set(near.tolist()) | set(excl.tolist()))

        d_target = 100 * (poisoned[t] - base_rnd[t])
        collateral = {c: 100 * (poisoned[c] - base_rnd[c]) for c in classes if c != t}
        worst_c = min(collateral, key=collateral.get)
        s_target = 100 * (selective[t] - base_rnd[t])
        s_collateral = {c: 100 * (selective[c] - base_rnd[c]) for c in classes if c != t}
        s_worst_c = min(s_collateral, key=s_collateral.get)
        fp = {"panel": pool_fp(P, pcls, Npan, POOL, t),
              "random": pool_fp(P, pcls, Nrnd, POOL, t),
              "nearest": pool_fp(P, pcls, Nnear, POOL, t),
              "selective": pool_fp(P, pcls, Nexcl, POOL, t)}
        p1 = d_target < collateral[worst_c]
        p1s = s_target < s_collateral[s_worst_c]
        p2 = fp["nearest"] <= fp["random"] + 0.01
        p2s = fp["selective"] <= fp["random"] + 0.01

        print(f"target {t}  (n={len(hi)})")
        print(f"  {'negative set':<22}{'target':>9}{'worst other':>9}{'  which':<32}{'pool FP':>9}")
        print(f"  {'panel only, ' + str(len(Npan)):<22}{base[t]:>9.1%}{'':>9}{'':<32}"
              f"{fp['panel']:>9.2%}")
        print(f"  {'+' + str(a.k) + ' random':<22}{base_rnd[t]:>9.1%}{'':>9}{'':<32}"
              f"{fp['random']:>9.2%}")
        print(f"  {'+' + str(a.k) + ' nearest':<22}{poisoned[t]:>9.1%}{poisoned[worst_c]:>9.1%}"
              f"  {worst_c:<30}{fp['nearest']:>9.2%}")
        print(f"  {'+' + str(a.k) + ' selective':<22}{selective[t]:>9.1%}"
              f"{selective[s_worst_c]:>9.1%}  {s_worst_c:<30}{fp['selective']:>9.2%}")
        print(f"  nearest:   target {d_target:+.1f}pt vs worst other {collateral[worst_c]:+.1f}pt"
              f"   P1 {'SUPPORTED' if p1 else 'REFUTED'}  P2 {'SUPPORTED' if p2 else 'REFUTED'}")
        print(f"  selective: target {s_target:+.1f}pt vs worst other "
              f"{s_collateral[s_worst_c]:+.1f}pt   P1 {'SUPPORTED' if p1s else 'REFUTED'}  "
              f"P2 {'SUPPORTED' if p2s else 'REFUTED'}   (Jaccard to nearest {jac:.3f})\n")

        res[t] = {"n": int(len(hi)), "target_panel": base[t], "target_random": base_rnd[t],
                  "target_nearest": poisoned[t], "target_delta_pts": d_target,
                  "collateral_pts": collateral, "worst_collateral_class": worst_c,
                  "worst_collateral_pts": collateral[worst_c],
                  "target_selective": selective[t], "selective_delta_pts": s_target,
                  "selective_collateral_pts": s_collateral,
                  "selective_worst_collateral_class": s_worst_c,
                  "selective_worst_collateral_pts": s_collateral[s_worst_c],
                  "selective_jaccard_to_nearest": jac,
                  "poisoned_lomo": poisoned, "selective_lomo": selective, "pool_fp": fp,
                  "P1_targeted_beats_collateral": bool(p1), "P2_fp_gives_no_signal": bool(p2),
                  "P1_selective_beats_collateral": bool(p1s),
                  "P2_selective_fp_gives_no_signal": bool(p2s)}

    ok = [t for t in res if res[t]["P1_targeted_beats_collateral"]]
    oks = [t for t in res if res[t]["P1_selective_beats_collateral"]]
    quiet = [t for t in res if res[t]["P2_fp_gives_no_signal"]]
    quiets = [t for t in res if res[t]["P2_selective_fp_gives_no_signal"]]
    verdict = (
        f"nearest-neighbour supplementation beats its own collateral damage on only "
        f"{len(ok)}/{len(res)} classes, so it is broad degradation rather than a targeted attack; "
        f"selecting for proximity to the target AND distance from every other class raises that to "
        f"{len(oks)}/{len(res)}. The pool false-positive rate stays within a point of the random "
        f"control on {len(quiet)}/{len(res)} and {len(quiets)}/{len(res)} respectively, so the "
        f"aggregate is a weak alarm either way. Largest single targeted drop: "
        f"{min(res, key=lambda t: res[t]['selective_delta_pts'])} at "
        f"{min(res[t]['selective_delta_pts'] for t in res):+.1f}pt under selective supplementation.")
    print(f"verdict: {verdict}")

    dest = V3 / f"reference_set_poisoning_{a.arm}.json"
    json.dump({"model": man["model"], "arm": a.arm, "k": a.k, "seeds": SEEDS,
               "panel_negatives": int(len(Npan)), "pool_n": int(len(POOL)),
               "contaminant_dropped": CONTAMINANT,
               "baseline_panel_lomo": base, "baseline_random_lomo": base_rnd,
               "per_target": res, "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
