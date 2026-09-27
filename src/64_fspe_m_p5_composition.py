#!/usr/bin/env python3
"""
64_fspe_m_p5_composition.py - P5's composition half: does dFSPE-M separate the panel from the benign
                              controls better than amino-acid composition alone?

P5 as frozen in section 4
-------------------------
    Supported if FSPE-M's protein-level separation exceeds that of an amino-acid-composition-only
    baseline by at least 0.05 AUROC. The composition baseline reaches 0.754 on the
    leave-one-mechanism-out panel, so it is not a straw man. A label-shuffled arm must return within
    0.05 of 0.5, and if it does not, the pipeline leaks and every other number here is void.

The shuffled arm is step 4's gate and was resolved in the fifth amendment: its leakage half passes,
its AUROC half has no power at n_benign = 4 and is reported without a pass/fail. This script supplies
the composition comparison, which is the other half and has never been run.

How the baseline is built, and why leave-one-out
-----------------------------------------------
The 20-dimensional amino-acid composition of each sequence, into a logistic regression, scored
**leave-one-out** so that no protein contributes to the model that scores it. Fitting on all 19 and
scoring the same 19 would report a fit, not a separation, at these n.

⚠️ **This comparison inherits the n_benign = 4 problem in full.** Section 4's 0.754 is from the
234-protein leave-one-mechanism-out panel; the AUROC here is over 15 x 4 = 60 pairs, whose null
standard deviation the fifth amendment measured at 0.167. A 0.05 margin between two such AUROCs is
far inside that noise. So the verdict is reported as **indeterminate rather than pass or fail**, for
the same reason and with the same arithmetic as the gate's AUROC half, and the numbers are given so a
reader can see how little they separate.

Usage:
    python src/64_fspe_m_p5_composition.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import load_positive_sequences, truncate_sequence  # noqa: E402

FSPE_M = ROOT / "results" / "fspe_m.json"
CONTROL_FASTA = ROOT / "data" / "sequences" / "benign_controls.fasta"
OUT = ROOT / "results" / "fspe_m_p5_composition.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
MARGIN = 0.05
N_PERM = 20000


def read_fasta(path):
    out, acc, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(buf)
            p = line[1:].split()[0].split("|")
            acc, buf = (p[1] if len(p) > 1 else line[1:].split()[0]), []
        elif acc:
            buf.append(line.strip())
    if acc:
        out[acc] = "".join(buf)
    return out


def composition(seq):
    n = max(len(seq), 1)
    return np.array([seq.count(a) / n for a in AA], float)


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg]))


def loo_scores(X, y):
    """Leave-one-out decision scores, so nothing is scored by a model it helped fit."""
    s = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        if len(set(y[m])) < 2:
            s[i] = np.nan
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        clf.fit(X[m], y[m])
        s[i] = clf.decision_function(X[i:i + 1])[0]
    return s


def main():
    d = json.load(open(FSPE_M))
    seqs = {}
    for sid, _desc, seq in load_positive_sequences():
        p = sid.split("|")
        seqs[p[1] if len(p) > 1 else sid] = seq
    seqs.update(read_fasta(CONTROL_FASTA))

    rows = []
    for r in d["panel"]:
        if r["dfspe_m"] is not None and r["acc"] in seqs:
            rows.append((r["acc"], 1, r["dfspe_m"]))
    for c in d["controls"]:
        if c["dfspe_m"] is not None and c["acc"] in seqs:
            rows.append((c["acc"], 0, c["dfspe_m"]))

    accs = [a for a, _, _ in rows]
    y = np.array([lab for _, lab, _ in rows])
    dfm = np.array([v for _, _, v in rows], float)
    X = np.vstack([composition(truncate_sequence(seqs[a], 1022)) for a in accs])

    comp = loo_scores(X, y)
    ok = ~np.isnan(comp)
    a_dfm = auroc(dfm[y == 1], dfm[y == 0])
    a_comp = auroc(comp[ok & (y == 1)], comp[ok & (y == 0)])
    delta = (a_dfm - a_comp) if (a_dfm is not None and a_comp is not None) else None

    # the null spread of an AUROC at these n, by permuting the labels: the same arithmetic the
    # fifth amendment used for the gate's AUROC half
    rng = np.random.default_rng(0)
    n1 = int((y == 1).sum())
    null = []
    for _ in range(N_PERM):
        v = rng.permutation(dfm)
        null.append(auroc(v[:n1], v[n1:]))
    null = np.array(null)

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statement": ("P5's composition half: dFSPE-M's panel-vs-control AUROC against a "
                      "leave-one-out amino-acid-composition baseline on the same proteins"),
        "n_panel": n1, "n_control": int((y == 0).sum()),
        "auroc_dfspe_m": a_dfm, "auroc_composition": a_comp, "difference": delta,
        "required_margin": MARGIN,
        "null_auroc_sd": float(null.std()),
        "margin_inside_noise": bool(delta is not None and abs(delta) < null.std()),
        "verdict": None,
        "section4_baseline_for_reference": 0.754,
        "baseline_scope_note": ("section 4's 0.754 is the composition baseline on the 234-protein "
                               "leave-one-mechanism-out panel, not on these 19 proteins; it is "
                               "quoted for reference and is not the comparator here"),
        "per_protein": [{"acc": a, "panel": int(lab), "dfspe_m": float(v),
                         "composition_score": (None if np.isnan(c) else float(c))}
                        for (a, lab, v), c in zip(rows, comp)],
    }
    res["verdict"] = (
        "INDETERMINATE: the required 0.05 margin is inside the sampling noise of an AUROC at "
        f"n_control = {res['n_control']}, whose null standard deviation is {null.std():.3f}. "
        "Reported without a pass or fail, on the same grounds as step 4's AUROC half."
        if res["margin_inside_noise"] else
        ("SUPPORTED" if delta is not None and delta >= MARGIN else "NOT SUPPORTED"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"n panel {res['n_panel']}  n control {res['n_control']}")
    print(f"AUROC dFSPE-M       {a_dfm:.4f}")
    print(f"AUROC composition   {a_comp:.4f}   (leave-one-out)")
    print(f"difference          {delta:+.4f}   required margin {MARGIN}")
    print(f"null AUROC sd       {null.std():.4f}  -> margin inside noise: "
          f"{res['margin_inside_noise']}")
    print(f"\n{res['verdict']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
