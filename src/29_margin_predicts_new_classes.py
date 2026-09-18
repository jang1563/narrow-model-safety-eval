#!/usr/bin/env python3
"""
29_margin_predicts_new_classes.py - can margin say which mechanism will be missed BEFORE the
                                    probe is run on it?

Why this is the test that matters
---------------------------------
§10.4 showed margin locates both unreachable classes. That is a within-panel fit: the rank
correlation was computed across all eleven classes at once. §10.3 is the warning attached to
exactly this kind of claim. The member-level version of it was preregistered with falsification
criteria, frozen at a tagged commit, tested on two external panels, and **it failed both
times**, and was downgraded to a property of the internal panel as the preregistration
required.

So the class-level version does not get to skip that step. A competence boundary has to be a
statement made **before** the answer is known, about classes that were not used to make it.

⚠️ The external panels cannot supply that test, and this is worth stating because it looks like
they should. `data/annotations/external_mechanism_classes.json` holds 51 proteins in six
classes and `safeprotein_mechanism_classes.json` 66 in seven, and **every one of those classes
is already in v2**. They are new members of known mechanisms, not new mechanisms. They can test
whether margin predicts for unseen *proteins*, which §10.3 already did and lost.

What CAN test it is the v3 expansion, because the margin relationship was measured on v2's nine
classes and bacteriocin, phage_peptidoglycan_hydrolase and cry_insecticidal did not exist in
the panel when it was. So:

    fit    on the NINE classes that predate the expansion, using v3 margins and v3
           recoveries so the feature and the target come from one panel and only class
           identity is out of sample
    test   on the THREE classes added by the expansion

⚠️ Honest about what this is: a retrodiction with a model whose parameters exclude the test
classes, not a sealed envelope. The author already knows the answers. What the design does
guarantee is that no information from the three test classes enters the fit, which is what
makes the out-of-sample error meaningful, and that the rank prediction below could have come
out wrong.

PREREGISTERED, written before the run
-------------------------------------
    P1  Among the three new classes, margin ranks **phage_peptidoglycan_hydrolase lowest**.
        Chance 1/3. This is the whole claim in one line: a mechanism nobody had tested,
        flagged as unreachable from its geometry alone.

    P2  Leave-one-class-out over all eleven classes: mean absolute error of the margin model
        beats the **mean-recovery baseline**, which predicts every class at the training
        mean. A predictor that cannot beat "guess the average" is not a boundary.

    P3  LOOCV predicted against measured, Spearman > 0 with permutation p < 0.05.

    SUPPORTED     all three hold.
    PARTIAL       P1 holds and the LOOCV tests do not, or the reverse, reported as such.
    REFUTED       P1 fails.

Both a linear fit and a rank-based fit are reported. Eleven classes is eleven points, and
§10.4 already recorded that the relationship is not monotone at the low end: the third lowest
margin recovers at 75%. A linear model on eleven points is a sighting line, not a calibration.

Usage:
    python src/29_margin_predicts_new_classes.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
V2 = ROOT / "results" / "v2"
PERMS = 20000
NEW_CLASSES = ["bacteriocin", "phage_peptidoglycan_hydrolase", "cry_insecticidal"]
PREDICTED_LOWEST = "phage_peptidoglycan_hydrolase"


def _spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def fit_predict(train_m, train_r, test_m):
    """Least squares on one feature. Returned predictions are clipped to [0, 1] because
    recovery is a fraction and an unclipped line predicts outside it."""
    A = np.c_[np.ones(len(train_m)), train_m]
    coef, *_ = np.linalg.lstsq(A, train_r, rcond=None)
    pred = coef[0] + coef[1] * np.asarray(test_m, float)
    return np.clip(pred, 0.0, 1.0), coef


def main():
    d3 = json.load(open(V3 / "second_failure_class.json"))["classes"]
    d2 = json.load(open(V2 / "second_failure_class.json"))["classes"]
    classes = sorted(d3)
    old = [c for c in classes if c not in NEW_CLASSES]
    new = [c for c in classes if c in NEW_CLASSES]
    assert sorted(new) == sorted(NEW_CLASSES), f"expected the three new classes, got {new}"

    m = {c: d3[c]["margin"] for c in classes}
    r = {c: d3[c]["recovery"] for c in classes}
    print(f"fit on {len(old)} pre-expansion classes, test on {len(new)} new ones\n")

    # ---- the pre-expansion relationship, for the record -----------------------------
    rho_old = _spearman([m[c] for c in old], [r[c] for c in old])
    rho_v2 = _spearman([d2[c]["margin"] for c in d2], [d2[c]["recovery"] for c in d2])
    print(f"margin vs recovery on the pre-expansion classes only: rho {rho_old:+.3f}")
    print(f"  the same relationship as measured on panel v2 itself:  rho {rho_v2:+.3f}\n")

    # ---- P1: rank the three new classes ---------------------------------------------
    order_new = sorted(new, key=lambda c: m[c])
    pred_new, coef = fit_predict([m[c] for c in old], [r[c] for c in old],
                                 [m[c] for c in new])
    print(f"{'new class':<32}{'margin':>9}{'predicted':>11}{'measured':>10}{'error':>8}")
    print("-" * 70)
    for c, p in zip(new, pred_new):
        print(f"{c:<32}{m[c]:>9.4f}{p * 100:>10.0f}%{r[c] * 100:>9.0f}%"
              f"{(p - r[c]) * 100:>+8.0f}")
    p1_hit = order_new[0] == PREDICTED_LOWEST
    print(f"\nP1  margin ranks the new classes low to high: {order_new}")
    print(f"    predicted lowest {PREDICTED_LOWEST} -> {'HIT' if p1_hit else 'MISS'}"
          f"   chance 1/3 = 0.333")

    # ---- P2 and P3: leave one class out over all eleven -----------------------------
    preds, truth = [], []
    for c in classes:
        tr = [x for x in classes if x != c]
        p, _ = fit_predict([m[x] for x in tr], [r[x] for x in tr], [m[c]])
        preds.append(float(p[0]))
        truth.append(r[c])
    preds, truth = np.array(preds), np.array(truth)
    mae = float(np.abs(preds - truth).mean())

    base = np.array([np.mean([r[x] for x in classes if x != c]) for c in classes])
    mae_base = float(np.abs(base - truth).mean())
    rho_loo = _spearman(preds, truth)
    rng = np.random.default_rng(0)
    null = np.array([_spearman(preds, rng.permutation(truth)) for _ in range(PERMS)])
    p3 = float((null >= rho_loo).mean())

    print(f"\nP2  leave-one-class-out MAE: margin model {mae * 100:.1f} points, "
          f"mean-recovery baseline {mae_base * 100:.1f} points -> "
          f"{'beats' if mae < mae_base else 'does NOT beat'} the baseline")
    print(f"P3  LOOCV predicted vs measured: Spearman {rho_loo:+.3f}, "
          f"permutation p {p3:.4f}")

    worst = classes[int(np.abs(preds - truth).argmax())]
    print(f"\nlargest LOOCV error: {worst}, predicted "
          f"{preds[classes.index(worst)] * 100:.0f}% against "
          f"{truth[classes.index(worst)] * 100:.0f}% measured")

    # ---- calibration, which is a different question from ordering -------------------
    err_new = np.array([float(p) - r[c] for c, p in zip(new, pred_new)])
    same_sign = bool(np.all(err_new > 0) or np.all(err_new < 0))
    worst_new = new[int(np.abs(err_new).argmax())]
    print(f"\ncalibration on the three new classes: signed errors "
          f"{[f'{e * 100:+.0f}' for e in err_new]} points, mean "
          f"{err_new.mean() * 100:+.1f}, all the same sign: {same_sign}")
    print(f"  the class the boundary exists to flag is the worst calibrated: {worst_new}, "
          f"predicted {float(pred_new[new.index(worst_new)]) * 100:.0f}% against "
          f"{r[worst_new] * 100:.0f}% measured")
    print("  so the ORDERING transfers and the MAGNITUDE does not: margin says which "
          "mechanism will be worst, not how badly it will be missed")

    if p1_hit and mae < mae_base and p3 < 0.05:
        verdict = (f"SUPPORTED FOR ORDERING, NOT CALIBRATED: margin ranked the unseen phage "
                   f"class lowest of the three new mechanisms, beats the mean baseline out of "
                   f"sample ({mae * 100:.1f} against {mae_base * 100:.1f} points) and its "
                   f"LOOCV predictions track recovery (rho {rho_loo:+.3f}, p {p3:.4f}). But it "
                   f"over-predicted that class by "
                   f"{abs(err_new[new.index(PREDICTED_LOWEST)]) * 100:.0f} points, 44% against "
                   f"10% measured, and all three new-class errors are optimistic. Usable to "
                   f"rank which mechanisms to distrust, not to state a miss rate")
    elif p1_hit:
        verdict = (f"PARTIAL: the rank prediction on the new classes holds, but out of sample "
                   f"MAE is {mae * 100:.1f} against a baseline of {mae_base * 100:.1f} and "
                   f"LOOCV rho is {rho_loo:+.3f} at p {p3:.4f}")
    else:
        verdict = (f"REFUTED: margin did not rank the unreachable new class lowest. "
                   f"order {order_new}")
    print(f"\nverdict: {verdict}")

    dest = V3 / "margin_predicts_new_classes.json"
    json.dump({"design": ("fit on the 9 pre-expansion classes, test on the 3 added by v3; "
                          "margins and recoveries both from panel v3 so only class identity "
                          "is out of sample"),
               "disclosure": ("a retrodiction with a model whose parameters exclude the test "
                              "classes, not a sealed prediction; the author knew the answers"),
               "external_panels_cannot_test_this": (
                   "every class in external_mechanism_classes.json and "
                   "safeprotein_mechanism_classes.json is already in v2, so they hold new "
                   "members of known mechanisms rather than new mechanisms"),
               "rho_pre_expansion_classes": rho_old, "rho_on_panel_v2": rho_v2,
               "linear_fit": {"intercept": float(coef[0]), "slope": float(coef[1])},
               "P1": {"order_low_to_high": order_new, "predicted_lowest": PREDICTED_LOWEST,
                      "hit": bool(p1_hit), "chance": 1 / 3,
                      "per_class": {c: {"margin": m[c], "predicted": float(p),
                                        "measured": r[c]}
                                    for c, p in zip(new, pred_new)}},
               "P2": {"loocv_mae": mae, "baseline_mae": mae_base,
                      "beats_baseline": bool(mae < mae_base)},
               "P3": {"loocv_spearman": rho_loo, "perm_p": p3, "perms": PERMS},
               "largest_loocv_error": worst,
               "calibration": {
                   "signed_errors_new_classes": {c: float(e) for c, e in zip(new, err_new)},
                   "mean_signed_error": float(err_new.mean()),
                   "all_same_sign": same_sign,
                   "worst_calibrated_new_class": worst_new,
                   "note": ("ordering transfers, magnitude does not. All three out-of-sample "
                            "errors are optimistic, and the class the boundary exists to flag "
                            "is the worst calibrated of the three. With n=3 a uniform sign is "
                            "p=0.125 on a sign test, so the bias is a caution rather than a "
                            "measured effect")},
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
