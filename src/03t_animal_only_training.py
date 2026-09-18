#!/usr/bin/env python3
"""
03t_animal_only_training.py - the clean version of the test §9.4 could not settle.

The question
------------
§2.4 established that target host is an uncontrolled confound: 51 of 80 positives
act on an animal host, 22 do not, and the two non-animal mechanism classes are
exactly the bottom two by recovery. The obvious follow-up is whether "non-animal
hazard" is a category the probe LEARNS from the non-animal positives it is given,
or whether those classes are simply hard.

§9.4 tried to answer a version of this and could not. Its without-AMR condition
removed only the 14 beta-lactamases, leaving contact-dependent inhibition and
colicin E2 in training, so the probe still had non-animal-target examples. This
removes every non-animal-target positive.

Why it is a 2x2 and not a single arm
------------------------------------
Training on animal-only removes 22 positives. A drop could therefore mean "the
non-animal examples mattered" OR "there was less training data". Two things
separate them:

    standard      train on every positive except the test class        (03b's arm)
    animal_only   train on ANIMAL-TARGET positives except the test class
    size_matched  train on a RANDOM subset of the same size as animal_only,
                  drawn from all positives except the test class

and both animal and non-animal classes are tested. The result is the INTERACTION:
does removing non-animal training data cost non-animal test classes more than it
costs animal test classes, beyond what losing the same number of random positives
costs?

One protocol throughout, 03b's: hold out 40% of the negatives, calibrate at the
95th percentile of the HELD-OUT negatives, 30 seeds. 25's mistake was comparing
figures from two different calibration protocols, so nothing here mixes them. The
external aminoglycoside class is scored the same way; it is never in training under
any condition.

PREREGISTERED, written before the run
-------------------------------------
Interaction = mean(standard - animal_only) over non-animal test classes
            - mean(standard - animal_only) over animal test classes

    SUPPORTED   interaction >= +15 points AND the size_matched condition accounts
                for less than 5 of it. Non-animal hazard is a category the probe
                learns from non-animal examples, which reframes beta-lactamase as
                a member of a category rather than a lone anomaly.

    REFUTED     interaction <= +5 points. The non-animal classes are hard on their
                own and having other non-animal positives in training does not help
                them.

    INCONCLUSIVE between the bounds, reported as such.

⚠️ Three non-animal test classes against six animal ones. The interaction is a
difference of two small means and the paired-by-seed interval is reported with it
rather than a bare point estimate.

Usage:
    python src/03t_animal_only_training.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
NEG_HOLDOUT_FRAC, SPEC = 0.40, 0.95
NONANIMAL_TARGETS = {"bacteria", "none_small_molecule", "plant", "other_nonanimal"}


def read_fasta(p):
    out, acc, seq = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(seq)
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        out[acc] = "".join(seq)
    return out


def embed_esm2(seqs, model_name="facebook/esm2_t33_650M_UR50D", max_len=1022):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    enc = tok([s[:max_len] for s in seqs], return_tensors="pt", padding=True,
              truncation=True, max_length=max_len + 2)
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1)
    return ((h * m).sum(1) / m.sum(1)).float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--external", default="data/sequences/amr_category_test.fasta")
    a = ap.parse_args()

    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    th = json.load(open(ROOT / "data/annotations/target_host_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    tgt = {e["fasta_id"]: e["target_host"] for e in th["proteins"]}
    accs = [r["acc"] for r in man["positive_rows"]]
    pcls = np.array([cls[x] for x in accs])
    ptgt = np.array([tgt[x] for x in accs])
    animal_pos = ptgt == "animal"
    print(f"panel: {len(P)} positives, {int(animal_pos.sum())} animal-target, "
          f"{int((~animal_pos).sum())} not")

    ext = read_fasta(ROOT / a.external)
    print(f"external class: {len(ext)} aminoglycoside-modifying enzymes")
    X_ext = embed_esm2(list(ext.values()))
    assert X_ext.shape[1] == P.shape[1]

    # test classes, grouped by whether their target is an animal
    internal = {}
    for C in sorted(set(pcls)):
        idx = np.where(pcls == C)[0]
        if len(idx) < 4 or C == "virulence_associated_non_toxin":
            continue
        ts = set(ptgt[idx])
        internal[C] = ("animal" if ts == {"animal"} else "non-animal", idx)
    tests = {C: (grp, P[idx], idx) for C, (grp, idx) in internal.items()}
    tests["aminoglycoside_external"] = ("non-animal", X_ext, np.array([], int))
    for C, (g, _, idx) in tests.items():
        print(f"  {C:<32}{len(idx) if len(idx) else len(ext):>3}  {g}")

    rng_master = np.random.default_rng(0)
    res = {C: {k: [] for k in ("standard", "animal_only", "size_matched")} for C in tests}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(N))
        cut = int(len(N) * NEG_HOLDOUT_FRAC)
        nte, ntr = perm[:cut], perm[cut:]
        for C, (grp, Xte, idx) in tests.items():
            held = set(idx.tolist())
            std = np.array([i for i in range(len(P)) if i not in held])
            ani = np.array([i for i in std if animal_pos[i]])
            k = len(ani)
            szm = rng_master.choice(std, k, replace=False) if k < len(std) else std
            for name, tri in (("standard", std), ("animal_only", ani), ("size_matched", szm)):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
                m.fit(np.vstack([P[tri], N[ntr]]),
                      np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
                thr = np.quantile(m.predict_proba(N[nte])[:, 1], SPEC)
                res[C][name].append(float((m.predict_proba(Xte)[:, 1] >= thr).mean()))

    print(f"\nflagged@95, {a.seeds} seeds, one protocol throughout")
    print(f"{'test class':<32}{'target':>12}{'standard':>10}{'animal':>9}"
          f"{'sizemat':>9}{'std-ani':>9}")
    print("-" * 81)
    rows = {}
    for C, (grp, _, _) in tests.items():
        s, an, sm = (float(np.mean(res[C][k])) for k in ("standard", "animal_only", "size_matched"))
        rows[C] = {"target": grp, "standard": s, "animal_only": an, "size_matched": sm,
                   "drop": s - an, "drop_size_matched": s - sm}
        print(f"{C:<32}{grp:>12}{s * 100:>9.1f}%{an * 100:>8.1f}%{sm * 100:>8.1f}%"
              f"{(s - an) * 100:>+8.1f}")

    non = [C for C in rows if rows[C]["target"] == "non-animal"]
    ani_c = [C for C in rows if rows[C]["target"] == "animal"]
    d_non = float(np.mean([rows[C]["drop"] for C in non]))
    d_ani = float(np.mean([rows[C]["drop"] for C in ani_c]))
    inter = d_non - d_ani
    s_non = float(np.mean([rows[C]["drop_size_matched"] for C in non]))
    s_ani = float(np.mean([rows[C]["drop_size_matched"] for C in ani_c]))
    inter_sm = s_non - s_ani

    # paired-by-seed interval on the interaction
    per_seed = []
    for i in range(a.seeds):
        dn = np.mean([res[C]["standard"][i] - res[C]["animal_only"][i] for C in non])
        da = np.mean([res[C]["standard"][i] - res[C]["animal_only"][i] for C in ani_c])
        per_seed.append(dn - da)
    per_seed = np.array(per_seed)
    bs = [np.mean(np.random.default_rng(i).choice(per_seed, len(per_seed)))
          for i in range(5000)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print(f"\nmean drop, non-animal test classes ({len(non)}): {d_non * 100:+.1f} points")
    print(f"mean drop, animal test classes ({len(ani_c)}):     {d_ani * 100:+.1f} points")
    print(f"INTERACTION: {inter * 100:+.1f} points, bootstrap 95% CI "
          f"[{lo * 100:+.1f}, {hi * 100:+.1f}]")
    print(f"size-matched control interaction: {inter_sm * 100:+.1f} points "
          f"(how much is explained by losing that many positives at random)")

    attributable = inter - inter_sm
    if inter >= 0.15 and inter_sm < 0.05:
        verdict = ("SUPPORTED: non-animal hazard is a category the probe learns from "
                   "non-animal examples")
    elif inter <= 0.05:
        verdict = ("REFUTED: removing non-animal training positives does not cost "
                   "non-animal test classes more than it costs animal ones")
    else:
        verdict = "INCONCLUSIVE by the preregistered bounds"
    if lo <= 0:
        verdict += "; ⚠️ the interaction's 95% interval includes 0"
    print(f"\nverdict: {verdict}")

    out = {"seeds": a.seeds, "protocol": "03b throughout: 40% negative holdout, "
           "threshold at the 95th percentile of held-out negatives",
           "n_animal_positives": int(animal_pos.sum()),
           "classes": rows,
           "drop_nonanimal": d_non, "drop_animal": d_ani,
           "interaction": inter, "interaction_ci95": [lo, hi],
           "interaction_size_matched": inter_sm,
           "interaction_attributable": attributable,
           "preregistered_supported_geq": 0.15, "preregistered_refuted_leq": 0.05,
           "ci_includes_zero": bool(lo <= 0), "verdict": verdict}
    p = V2 / "animal_only_training.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
