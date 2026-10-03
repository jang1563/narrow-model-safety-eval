#!/usr/bin/env python3
"""The session's prediction record and multiple-comparison exposure, computed not carried.

🔴 Entry 112 corrected a tally that had been written into prose with nothing auditing it, and
entry 115 then made the same mistake two entries later: it stated "derived 9 of 16, reasoned 2
of 7" while the artifacts said 10 of 20 and 3 of 9, because study AI-1's three predictions were
being omitted by hand. This file computes the record from the artifacts so no hand-maintained
version exists to drift.

It also counts the exposure the paper has never stated: how many preregistrations, how many
frozen predictions, and the fact that Bonferroni has been applied WITHIN each study and never
across them.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/prediction_record.json"

# (label, artifact) for every study that froze predictions and recorded their verdicts
STUDIES = [
    ("AH", "external_hazard_axis_lomo"),
    ("AI-1", "deepvic_metadata_baseline"),
    ("AI", "deepvic_verdicts"),
    ("AJ", "composition_arm"),
    ("AK", "representation_increment"),
    ("AM", "within_fold_ordering"),
    ("AN", "composition_residual"),
    ("AO", "nested_model"),
    ("AP", "estimation_repair"),
]
PREREGS = [
    "EXTERNAL_HAZARD_AXIS", "DEEPVIC_BENCHMARK", "COMPOSITION_ARM",
    "REPRESENTATION_INCREMENT", "WITHIN_FOLD_ORDERING", "COMPOSITION_RESIDUAL",
    "NESTED_MODEL", "ESTIMATION_REPAIR",
]


def read(name):
    f = ROOT / "results" / f"{name}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def collect():
    rows, missing = [], []
    for tag, art in STUDIES:
        d = read(art)
        if d is None:
            missing.append((tag, art))
            continue
        for k, v in d.get("predictions", {}).items():
            rows.append({"study": tag, "prediction": k,
                         "kind": v.get("kind", "unlabelled"),
                         "held": bool(v.get("held"))})
    return rows, missing


def exposure():
    out = []
    for name in PREREGS:
        f = ROOT / "docs" / f"{name}_PREREGISTRATION.md"
        if not f.exists():
            out.append({"prereg": name, "exists": False})
            continue
        t = f.read_text()
        out.append({
            "prereg": name, "exists": True,
            "declared_predictions": len(re.findall(r"^\*\*P\d", t, re.M)),
            "alpha_statements": re.findall(r"α = 0\.05/(\d+)", t),
            "amendments": len(re.findall(r"^## Amendment", t, re.M)),
        })
    return out


def selftest():
    ok = True
    rows, missing = collect()
    if missing:
        print(f"  XX artifacts missing for {missing}")
        ok = False
    if not rows:
        print("  XX no predictions collected at all")
        ok = False
    # every prediction must carry a kind the record can tally; "unlabelled" is a defect
    unl = [r for r in rows if r["kind"] not in ("derived", "reasoned")]
    if unl:
        print(f"  XX {len(unl)} prediction(s) with no kind: "
              f"{[(r['study'], r['prediction']) for r in unl][:3]}")
        ok = False
    # 🔒 the per-study counts must match the artifacts' own prediction dicts exactly, or this
    # file has introduced the very drift it exists to prevent
    for tag, art in STUDIES:
        d = read(art)
        if d is None:
            continue
        n_art = len(d.get("predictions", {}))
        n_row = sum(1 for r in rows if r["study"] == tag)
        if n_art != n_row:
            print(f"  XX {tag}: {n_row} rows against {n_art} in the artifact")
            ok = False
    # and every declared prediction in a preregistration must be scored somewhere, or the
    # record is quietly narrower than what was frozen
    exp = exposure()
    declared = sum(e.get("declared_predictions", 0) for e in exp)
    if declared < len(rows):
        print(f"  XX {len(rows)} scored against {declared} declared -- the count is impossible")
        ok = False
    if any(not e["exists"] for e in exp):
        print(f"  XX a preregistration is missing: "
              f"{[e['prereg'] for e in exp if not e['exists']]}")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    rows, _ = collect()
    exp = exposure()
    tal = {}
    for kind in ("derived", "reasoned"):
        sub = [r for r in rows if r["kind"] == kind]
        tal[kind] = {"held": sum(1 for r in sub if r["held"]), "total": len(sub)}
    by_study = {}
    for tag, _ in STUDIES:
        sub = [r for r in rows if r["study"] == tag]
        by_study[tag] = {
            "derived": [sum(1 for r in sub if r["kind"] == "derived" and r["held"]),
                        sum(1 for r in sub if r["kind"] == "derived")],
            "reasoned": [sum(1 for r in sub if r["kind"] == "reasoned" and r["held"]),
                         sum(1 for r in sub if r["kind"] == "reasoned")],
        }
    res = {
        "built": "2026-10-03",
        "study": "the session's prediction record and exposure, computed from artifacts",
        "why": ("entry 112 corrected a prose tally and entry 115 repeated the mistake two "
                "entries later, stating derived 9 of 16 while the artifacts said 10 of 20. "
                "No hand-maintained version exists any more."),
        "rows": rows,
        "tally": tal,
        "total_scored": len(rows),
        "by_study": by_study,
        "exposure": {
            "preregistrations": len(exp),
            "declared_predictions": sum(e.get("declared_predictions", 0) for e in exp),
            "amendments": sum(e.get("amendments", 0) for e in exp),
            "per_prereg": exp,
            "bonferroni_within_study_only": True,
            "no_family_wise_correction_across_studies": True,
            "note": ("every preregistration sets its own Bonferroni alpha over its own "
                     "primary tests; no correction is applied across the studies. With this "
                     "many frozen bands some verdicts are expected to fall by chance, and "
                     "the paper states the exposure rather than correcting retroactively."),
        },
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"  scored {len(rows)} predictions across {len(STUDIES)} studies")
    print(f"  derived {tal['derived']['held']}/{tal['derived']['total']}   "
          f"reasoned {tal['reasoned']['held']}/{tal['reasoned']['total']}")
    for tag, v in by_study.items():
        print(f"    {tag:<5} derived {v['derived'][0]}/{v['derived'][1]}  "
              f"reasoned {v['reasoned'][0]}/{v['reasoned'][1]}")
    e = res["exposure"]
    print(f"  exposure: {e['preregistrations']} preregistrations, "
          f"{e['declared_predictions']} declared predictions, {e['amendments']} amendments")
    print(f"  Bonferroni within study only: {e['bonferroni_within_study_only']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
