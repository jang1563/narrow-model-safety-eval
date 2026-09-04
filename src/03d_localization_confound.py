#!/usr/bin/env python3
"""
03d_localization_confound.py - how much of the probe's separation is localization?

Background
----------
With organism held constant the embeddings separate secreted from cytoplasmic
negatives at AUROC 1.000, and the two label axes are correlated: most positives
are exported, most negatives are not. `pathogen_matched` controls organism but
not localization; `secreted_only` controls localization but not organism. This
script measures the confound instead of arguing about it, using the independent
UniProt annotation written by 02c.

What it reports
---------------
  cross-tab          localization against the hazard label, plus the agreement
                     rate. This is the localization analogue of the 67% organism
                     agreement already reported by 03b.
  localization_auroc can the embedding predict EXPORTED vs NOT, hazard ignored,
                     across all 166 proteins. High means the axis is strongly
                     encoded, which is why the stratified numbers below matter.
  stratified         baseline AUROC computed SEPARATELY inside the exported
                     stratum and inside the non-exported stratum. If hazard were
                     only a proxy for localization, separation should collapse
                     once localization is held constant.
  doubly_controlled  positives against pathogen-derived negatives, restricted to
                     one localization stratum, so organism AND localization are
                     both held constant. Reported only where n allows.

Two label definitions are used side by side because neither is beyond argument:
  exported          localization in {secreted, cell_surface, periplasm, membrane}
  signal_peptide    UniProt Signal feature present. Curator-independent, but
                    misses non-classical secretion.

Run after 02b and 02c. Uses cached embeddings; takes seconds.

Usage:
    python src/03d_localization_confound.py [--tag smoke150M]
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
EXPORTED = {"secreted", "cell_surface", "periplasm", "membrane"}
MIN_PER_CLASS = 8  # below this a 5-fold CV AUROC is not worth printing


def cv_auroc(X, y, seed=0):
    if min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS:
        return None, None
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    s = cross_val_score(
        pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc"
    )
    return float(s.mean()), float(s.std())


def show(label, X, y, note=""):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    m, s = cv_auroc(X, y)
    if m is None:
        print(f"{label:<34}{'n too small':>16}   ({npos} vs {nneg}) {note}")
        return None
    print(f"{label:<34}AUROC {m:.3f} +/- {s:.3f}   ({npos} vs {nneg}) {note}")
    return [m, s, npos, nneg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    tag = ap.parse_args().tag
    suf = f"_{tag}" if tag else ""

    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    ann = json.load(open(ROOT / "data/annotations/localization_v2.json"))["proteins"]
    panel = json.load(open(ROOT / "data/sequences/panel_v2_manifest.json"))

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    neg_acc = [r["acc"] for r in man["negative_rows"]]
    assert len(pos_acc) == P.shape[0] and len(neg_acc) == N.shape[0]

    ids = pos_acc + neg_acc
    X = np.vstack([P, N])
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]  # hazard
    loc = np.array([ann[i]["localization"] for i in ids])
    exp = np.array([ann[i]["localization"] in EXPORTED for i in ids])
    sig = np.array([ann[i]["has_signal_peptide"] for i in ids])
    pdc_of = {n["acc"]: bool(n.get("pathogen_derived_control")) for n in panel["negatives"]}
    pdc = np.array([pdc_of.get(i, False) for i in ids])

    out = {
        "model": man["model"],
        "tag": tag or None,
        "n_positive": int(len(P)),
        "n_negative": int(len(N)),
    }

    # ---- 1. how correlated are the two label axes -------------------------
    print("cross-tab: rows = hazard, cols = localization axis\n")
    for name, axis in (("exported", exp), ("signal_peptide", sig)):
        a = float(np.mean(axis == (y == 1)))
        tp = int(((y == 1) & axis).sum())
        fn = int(((y == 1) & ~axis).sum())
        fp = int(((y == 0) & axis).sum())
        tn = int(((y == 0) & ~axis).sum())
        print(
            f"  {name:<16} positive {tp:>3} yes / {fn:>3} no      "
            f"negative {fp:>3} yes / {tn:>3} no      agreement with hazard {a:.0%}"
        )
        out[f"{name}_agreement_with_hazard"] = a
        out[f"{name}_crosstab"] = {"pos_yes": tp, "pos_no": fn, "neg_yes": fp, "neg_no": tn}
    print()

    # ---- 2. is the axis itself encoded ------------------------------------
    out["localization_axis_auroc"] = show(
        "localization axis (hazard ignored)", X, exp.astype(int), "<- exported vs not, all proteins"
    )
    out["signal_axis_auroc"] = show(
        "signal-peptide axis (hazard ignored)", X, sig.astype(int), "<- signal vs not, all proteins"
    )
    print()

    # ---- 3. hazard separation INSIDE each localization stratum -------------
    print("hazard separation with localization held constant")
    strata = {}
    for nm, mask in (
        ("exported", exp),
        ("not_exported", ~exp),
        ("signal_peptide", sig),
        ("no_signal_peptide", ~sig),
    ):
        strata[nm] = show(f"  baseline | {nm}", X[mask], y[mask])
    out["stratified_baseline"] = strata
    print()

    # ---- 4. organism AND localization both controlled ----------------------
    print("organism AND localization both held constant")
    dbl = {}
    for nm, mask in (
        ("exported", exp),
        ("not_exported", ~exp),
        ("signal_peptide", sig),
        ("no_signal_peptide", ~sig),
    ):
        keep = mask & ((y == 1) | pdc)  # positives + pathogen-derived negatives only
        dbl[nm] = show(f"  pathogen-matched | {nm}", X[keep], y[keep])
    out["doubly_controlled"] = dbl

    # ---- 5. sensitivity: proteins with no location annotation dropped ------
    # `exported` counts an unannotated protein as not-exported, and most of the
    # not-exported positives are unannotated rather than verified cytoplasmic.
    # If the stratified result only holds because of those, it is not a result.
    known = loc != "unknown"
    print(
        f"\nsensitivity: drop unannotated localization "
        f"({int((~known & (y == 1)).sum())} positives, {int((~known & (y == 0)).sum())} negatives)"
    )
    sens = {}
    for nm, mask in (("exported", exp & known), ("not_exported", ~exp & known)):
        sens[f"baseline_{nm}"] = show(f"  baseline | {nm}", X[mask], y[mask])
    for nm, mask in (("exported", exp & known), ("not_exported", ~exp & known)):
        keep = mask & ((y == 1) | pdc)
        sens[f"pathogen_matched_{nm}"] = show(f"  pathogen-matched | {nm}", X[keep], y[keep])
    out["sensitivity_known_localization_only"] = sens
    out["n_unknown_localization"] = {
        "positive": int((~known & (y == 1)).sum()),
        "negative": int((~known & (y == 0)).sum()),
    }

    p = V2 / f"localization_confound{suf}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    print(
        "\nReading guide: if hazard separation survives INSIDE a localization stratum, the probe is"
    )
    print(
        "not merely reading localization. The doubly controlled rows are the strongest form of that"
    )
    print("test, since organism is held constant there as well.")


if __name__ == "__main__":
    main()
