#!/usr/bin/env python3
"""
03b_leave_one_mechanism_out.py

Question: if a hazardous protein uses a mechanism the probe has never seen,
is it still flagged, or is it confidently cleared? This is the "molecule that
is not on the list" case, tested by holding out an entire mechanism class.

Design notes that make the numbers defensible
---------------------------------------------
* Threshold, not AUROC, for small classes. Several classes have n=3, where AUROC
  can take only a handful of values. We instead calibrate an operating threshold
  on TRAINING negatives and report (a) each held-out protein's probability and
  (b) the fraction flagged. AUROC is reported only when the held-out class has
  n >= 7.
* Negatives are split too, and the threshold is calibrated ON THE HELD-OUT
  negatives. A first pass calibrated on TRAINING negatives, which the model has
  already seen; they scored near zero, the 95th percentile landed at about 0.026,
  and essentially any unseen protein cleared it. Held-out FPR came out at 35-51%
  instead of 5%, which is how the error showed. Calibrating on held-out negatives
  makes FPR 5% by construction, so "flagged" means: caught at the operating point
  where 5% of UNSEEN benign proteins get through. 40% of the negatives are held
  out to keep the quantile estimate usable, averaged over five seeds.
* Scaler and PCA are fit on training folds only.

Controls run alongside the main experiment
------------------------------------------
All counts are read from the panel at run time; none are hard-coded here.

  baseline           all positives vs all negatives, 5-fold CV
  pca50              same, on 50 PCA components (d >> n check)
  secreted_only      positives vs the condition-matched secreted negatives
  virulence_holdout  generalization from toxins to non-toxin virulence factors
  localization       secreted vs housekeeping, LAB-STRAIN negatives only, so
                     organism is held constant. The lab-strain filter matters:
                     once pathogen-derived negatives were added, an unfiltered
                     secreted-vs-housekeeping split stopped being organism
                     controlled and became localization plus organism.
  pathogen_matched   positives vs benign proteins from the SAME pathogens. This
                     is the load-bearing control: it measures hazard signal net
                     of provenance rather than arguing the confound away.
  provenance         lab-strain origin vs not, across all proteins, ignoring the
                     hazard label. Positives span many organisms while a large
                     share of negatives come from two lab strains, so organism
                     and hazard labels are correlated; the run prints the exact
                     agreement. A high AUROC here means the panel cannot separate
                     hazard signal from provenance signal on its own, which is
                     why pathogen_matched exists. The leave-one-mechanism-out
                     CONTRAST is less exposed, since every holdout class is
                     pathogen-derived.

Usage:
    python src/03b_leave_one_mechanism_out.py [--tag smoke150M]
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
SEEDS = [0, 1, 2, 3, 4]
NEG_HOLDOUT_FRAC = 0.40


def clf(pca=None):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, random_state=0))
    steps.append(LogisticRegression(max_iter=5000, C=1.0))
    return make_pipeline(*steps)


def cv_auroc(X, y, pca=None, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = cross_val_score(clf(pca), X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def threshold_at_specificity(scores_neg, spec):
    """Score above which only (1-spec) of the supplied negatives fall.

    Must be given HELD-OUT negatives. Training negatives are already fit and
    score near zero, which collapses the threshold and makes it meaningless.
    """
    return float(np.quantile(scores_neg, spec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    tag = ap.parse_args().tag
    suf = f"_{tag}" if tag else ""

    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    panel = json.load(open(ROOT / "data/sequences/panel_v2_manifest.json"))

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    neg_acc = [r["acc"] for r in man["negative_rows"]]
    assert len(pos_acc) == P.shape[0] and len(neg_acc) == N.shape[0]

    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    name_of = {p["fasta_id"]: p["short_name"] for p in mech["proteins"]}
    holdout_classes = set(mech["holdout_eligible_classes"])
    # Every panel lookup goes through the accession, never through list position.
    # Positional indexing into panel[...] silently breaks whenever the panel is
    # re-ordered or extended, which is the same failure 02b was written to prevent.
    block_of = {n["acc"]: n["block"] for n in panel["negatives"]}
    pdc_of = {n["acc"]: bool(n.get("pathogen_derived_control")) for n in panel["negatives"]}
    lab_of = {n["acc"]: bool(n.get("lab_strain")) for n in panel["negatives"]}
    lab_of.update({p["acc"]: bool(p.get("lab_strain")) for p in panel["positives"]})
    secreted_idx = np.array(
        [i for i, a in enumerate(neg_acc) if block_of.get(a) == "secreted_cellwall"]
    )

    pos_cls = np.array([cls_of.get(a, "UNMAPPED") for a in pos_acc])
    X = np.vstack([P, N])
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    out = {
        "model": man["model"],
        "tag": tag or None,
        "n_positive": int(len(P)),
        "n_negative": int(len(N)),
        "embedding_dim": int(P.shape[1]),
    }

    print(f"model={man['model']}  positives={len(P)}  negatives={len(N)}  dim={P.shape[1]}\n")

    # ---- controls -----------------------------------------------------------
    m, s = cv_auroc(X, y)
    out["baseline_auroc"] = [m, s]
    print(f"baseline           AUROC {m:.3f} +/- {s:.3f}   ({len(P)} vs {len(N)})")

    m, s = cv_auroc(X, y, pca=50)
    out["pca50_auroc"] = [m, s]
    print(f"pca50              AUROC {m:.3f} +/- {s:.3f}   (d >> n check)")

    Xs = np.vstack([P, N[secreted_idx]])
    ys = np.r_[np.ones(len(P)), np.zeros(len(secreted_idx))]
    m, s = cv_auroc(Xs, ys)
    out["secreted_only_auroc"] = [m, s, int(len(secreted_idx))]
    print(
        f"secreted_only      AUROC {m:.3f} +/- {s:.3f}   ({len(P)} vs {len(secreted_idx)} secreted)"
    )

    house_idx = np.array([i for i in range(len(neg_acc)) if i not in set(secreted_idx.tolist())])
    # Organism-controlled localization. Restrict BOTH sides to the lab-strain
    # negatives. Before the pathogen-derived block was added, every negative came
    # from the same two lab strains, so this comparison was organism-controlled by
    # construction. It no longer is: most housekeeping negatives are now
    # pathogen-derived, which would turn this into a localization-plus-organism
    # comparison. Filtering restores what the control was built to measure.
    sec_lab = np.array([i for i in secreted_idx.tolist() if lab_of.get(neg_acc[i])])
    hou_lab = np.array([i for i in house_idx.tolist() if lab_of.get(neg_acc[i])])
    Xl = np.vstack([N[sec_lab], N[hou_lab]])
    yl = np.r_[np.ones(len(sec_lab)), np.zeros(len(hou_lab))]
    m, s = cv_auroc(Xl, yl)
    out["localization_auroc"] = [m, s, int(len(sec_lab)), int(len(hou_lab))]
    print(
        f"localization       AUROC {m:.3f} +/- {s:.3f}   "
        f"({len(sec_lab)} secreted vs {len(hou_lab)} housekeeping, lab strains only)"
    )
    print("                   ^ localization signal with organism held constant")

    pth = np.array([i for i, a in enumerate(neg_acc) if pdc_of.get(a)])
    if len(pth):
        Xp = np.vstack([P, N[pth]])
        yp = np.r_[np.ones(len(P)), np.zeros(len(pth))]
        m, s = cv_auroc(Xp, yp)
        out["pathogen_matched_auroc"] = [m, s, int(len(pth))]
        print(
            f"pathogen_matched   AUROC {m:.3f} +/- {s:.3f}   "
            f"({len(P)} vs {len(pth)} benign proteins from the SAME pathogens)"
        )
        print("                   ^ organism controlled: hazard signal net of provenance")

    lab_pos = np.array([lab_of[a] for a in pos_acc])
    lab_neg = np.array([lab_of[a] for a in neg_acc])
    ylab = np.r_[lab_pos, lab_neg].astype(int)
    agree = float(np.mean((ylab == 0) == (y == 1)))
    m, s = cv_auroc(X, ylab)
    out["provenance_auroc"] = [m, s]
    out["organism_label_agreement_with_hazard"] = agree
    print(
        f"provenance         AUROC {m:.3f} +/- {s:.3f}   (lab-strain origin, hazard label ignored)"
    )
    print(f"                   organism label agrees with hazard label on {agree:.0%} of proteins")
    print(
        "                   ^ high AUROC + high agreement = baseline cannot be attributed to hazard\n"
    )

    # ---- leave-one-mechanism-out -------------------------------------------
    targets = sorted(
        {c for c in pos_cls if c in holdout_classes} | {"virulence_associated_non_toxin"}
    )
    res = {}
    print(
        f"{'class':<34}{'n':>3}  {'flagged@95':>11} {'flagged@99':>11}  {'AUROC':>7}  probabilities"
    )
    print("-" * 118)
    for C in targets:
        hi = np.where(pos_cls == C)[0]
        if len(hi) == 0:
            continue
        rec = {
            "n": int(len(hi)),
            "members": [name_of.get(pos_acc[i], pos_acc[i]) for i in hi],
            "holdout_eligible": C in holdout_classes,
            "per_seed": [],
        }
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            nperm = rng.permutation(len(N))
            ncut = int(len(N) * NEG_HOLDOUT_FRAC)
            nte, ntr = nperm[:ncut], nperm[ncut:]
            tri = np.array([i for i in range(len(P)) if i not in set(hi.tolist())])
            Xtr = np.vstack([P[tri], N[ntr]])
            ytr = np.r_[np.ones(len(tri)), np.zeros(len(ntr))]
            model = clf().fit(Xtr, ytr)
            s_ho = model.predict_proba(P[hi])[:, 1]
            s_nte = model.predict_proba(N[nte])[:, 1]
            # calibrate on HELD-OUT negatives; see module docstring
            t95 = threshold_at_specificity(s_nte, 0.95)
            t99 = threshold_at_specificity(s_nte, 0.99)
            rec["per_seed"].append(
                {
                    "seed": seed,
                    "t95": t95,
                    "t99": t99,
                    "probs": [round(float(v), 3) for v in s_ho],
                    "flagged_95": float((s_ho >= t95).mean()),
                    "flagged_99": float((s_ho >= t99).mean()),
                    "fpr_heldout_95": float(
                        (s_nte >= t95).mean()
                    ),  # ~0.05 by construction, sanity check
                    "auroc": (
                        float(
                            roc_auc_score(
                                np.r_[np.ones(len(s_ho)), np.zeros(len(s_nte))], np.r_[s_ho, s_nte]
                            )
                        )
                        if len(hi) >= 7
                        else None
                    ),
                }
            )
        f95 = np.mean([r["flagged_95"] for r in rec["per_seed"]])
        f99 = np.mean([r["flagged_99"] for r in rec["per_seed"]])
        aus = [r["auroc"] for r in rec["per_seed"] if r["auroc"] is not None]
        rec["flagged_95_mean"], rec["flagged_99_mean"] = float(f95), float(f99)
        rec["auroc_mean"] = float(np.mean(aus)) if aus else None
        res[C] = rec
        au = f"{rec['auroc_mean']:.3f}" if rec["auroc_mean"] is not None else "n<7"
        pr = ", ".join(f"{v:.2f}" for v in rec["per_seed"][0]["probs"][:6])
        print(f"{C:<34}{len(hi):>3}  {f95:>10.0%} {f99:>11.0%}  {au:>7}  {pr}")
    out["leave_one_mechanism_out"] = res

    V2.mkdir(parents=True, exist_ok=True)
    p = V2 / f"lomo_results{suf}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    print(
        "\nReading guide: 'flagged@95' is the fraction of held-out class members scoring above the"
    )
    print(
        "threshold that lets 5% of training negatives through. A class the probe never saw that is"
    )
    print("still flagged means mechanism generalises; one that is cleared means it does not.")


if __name__ == "__main__":
    main()
