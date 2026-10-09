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

# 🔴 These were two HAND-MAINTAINED lists, and study AQ's four predictions were silently
# absent from the record because neither list was updated when it ran. The "every declared
# prediction is scored somewhere" guard passed, because it compared one stale list against the
# other -- consistent, and both incomplete. This file prevented drift between the artifact and
# the paper and not between the artifact and reality. Both lists are now DISCOVERED.
#
# A label is still needed for display, so artifacts not in this map are reported under their
# own filename rather than skipped.
LABELS = {
    "external_hazard_axis_lomo": "AH", "deepvic_metadata_baseline": "AI-1",
    "deepvic_verdicts": "AI", "composition_arm": "AJ",
    "representation_increment": "AK", "within_fold_ordering": "AM",
    "composition_residual": "AN", "nested_model": "AO",
    "estimation_repair": "AP", "fair_comparison": "AQ",
}


def discover_studies():
    """Every results/*.json carrying a `predictions` dict, found rather than listed."""
    out = []
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get("predictions"), dict) and d["predictions"]:
            out.append((LABELS.get(f.stem, f.stem), f.stem))
    return out


def discover_preregs():
    """Preregistrations that DECLARE numbered predictions, found rather than listed.

    ⚠️ The repository holds forty preregistration documents; only the ones using the `**Pn`
    form declare numbered predictions, and only those contribute to the exposure. A first
    version counted all forty and the cross-guard below fired on the difference.
    """
    out = []
    for f in sorted((ROOT / "docs").glob("*_PREREGISTRATION.md")):
        if re.findall(r"^\*\*P\d", f.read_text(), re.M):
            out.append(f.name[: -len("_PREREGISTRATION.md")])
    return out


def scored_preregs():
    """Which preregistrations a scored artifact points back at.

    🔒 No hand-maintained map: every study artifact carries its own `preregistration` path,
    so the link from a scored result to the document that froze it is in the data already.
    """
    seen = set()
    for _, art in discover_studies():
        d = read(art) or {}
        q = d.get("preregistration", "")
        m = re.search(r"([A-Z_]+)_PREREGISTRATION\.md", q)
        if m:
            seen.add(m.group(1))
    return seen


def read(name):
    f = ROOT / "results" / f"{name}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


# 🔒 Preregistrations that DECLARE predictions but have no scored artifact yet. Each needs a
# REASON, because the default reading of an unscored preregistration is that a study ran and
# its verdicts never reached the record -- which is how study AQ went missing. A name added
# here is a statement that the design is frozen and the run has not happened; a name that
# should NOT be here fails the selftest loudly. ⚠️ Adding a name is the only way to silence
# the check, which is deliberate: it makes the omission a written claim.
UNRUN = {
    "MUTATION_EXTENSION": "a frozen design for work not yet run, per README",
    "GENERALITY_GRID": ("study AT, frozen 2026-10-06; its six cells require a second text "
                        "encoder, two image encoders and ProtT5 on DeepVIC, and no cell is "
                        "scored until all six land"),
    "FINETUNE_RECOVERY": ("study AY, frozen 2026-10-08; amendment 2 added two mean-pooling "
                          "recipes after P3's failure localized a head confound, so the "
                          "five-recipe array is still fitting on the cluster. Remove this "
                          "entry when results/finetune_recovery.json lands"),
}



def reconcile(exp, rows):
    """declared = scored + unrun + withdrawn - double-scored, as an identity, not a literal.

    🔒 Returns the pieces AND whether they balance, so a claim can assert the balance and a
    reader can see which term moved. DEEPVIC_BENCHMARK is scored by two artifacts whose
    prediction numbers overlap, so it contributes more rows than it declares; that surplus is
    the only reason the naive subtraction fails.
    """
    by = {e["prereg"]: e.get("declared_predictions", 0) for e in exp if e["exists"]}
    declared = sum(by.values())
    scored = sum(1 for r in rows if r["kind"] in ("derived", "reasoned"))
    withdrawn = sum(1 for r in rows if r["kind"] == "withdrawn")
    unrun = sum(by.get(k, 0) for k in UNRUN if k in by)
    # rows a preregistration holds beyond what it declared, summed over preregistrations
    rows_per = {}
    for tag, art in discover_studies():
        d = read(art)
        if d is None:
            continue
        pr = Path(d.get("preregistration", "")).name.replace("_PREREGISTRATION.md", "")
        rows_per[pr] = rows_per.get(pr, 0) + len(d.get("predictions", {}))
    surplus = sum(max(0, n - by.get(pr, 0)) for pr, n in rows_per.items())
    return {"declared": declared, "scored": scored, "withdrawn": withdrawn,
            "unrun": unrun, "double_scored_surplus": surplus,
            "balances": declared == scored + unrun + withdrawn - surplus,
            "identity": ("declared = scored + unrun + withdrawn - double_scored_surplus")}


def collect():
    rows, missing = [], []
    for tag, art in discover_studies():
        d = read(art)
        if d is None:
            missing.append((tag, art))
            continue
        for k, v in d.get("predictions", {}).items():
            h = v.get("held")
            kind = v.get("kind", "unlabelled")
            # A WITHDRAWN prediction counts as neither held nor missed, so `held` stays
            # None rather than being coerced to False -- `bool(None)` is False, which
            # reads as a MISS in the artifact and is what the preregistration forbids.
            rows.append({"study": tag, "prediction": k, "kind": kind,
                         "held": None if h is None else bool(h)})
    for r in rows:
        if r["kind"] == "withdrawn" and r["held"] is not None:
            raise SystemExit(f"XX withdrawn prediction {r['study']}/{r['prediction']} "
                             f"carries held={r['held']!r}; it must be null")
        if r["kind"] in ("derived", "reasoned") and r["held"] is None:
            raise SystemExit(f"XX {r['kind']} prediction {r['study']}/{r['prediction']} "
                             f"has held=null; only `withdrawn` may be unscored")
    return rows, missing


def exposure():
    out = []
    for name in discover_preregs():
        f = ROOT / "docs" / f"{name}_PREREGISTRATION.md"
        if not f.exists():
            out.append({"prereg": name, "exists": False})
            continue
        t = f.read_text()
        out.append({
            "prereg": name, "exists": True,
            # 🔴 Was `len(re.findall(r"^\*\*P\d", t, re.M))`, which counted LINES and so
            # counted an amendment's restatement ("**P1 and P2 are unchanged.") as a further
            # declaration -- CROSS_MODALITY read 5 against its actual P1-P4. Distinct
            # prediction NUMBERS is the honest reading: a restatement adds no declaration,
            # and an amendment that adds P5 still counts once.
            "declared_predictions": len(set(re.findall(r"^\*\*P(\d+)", t, re.M))),
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
    # `withdrawn` is a third legitimate kind: a prediction whose referent an amendment
    # removed. It carries no verdict by design, and `collect()` already enforces that.
    unl = [r for r in rows if r["kind"] not in ("derived", "reasoned", "withdrawn")]
    if unl:
        print(f"  XX {len(unl)} prediction(s) with no kind: "
              f"{[(r['study'], r['prediction']) for r in unl][:3]}")
        ok = False
    # 🔒 the per-study counts must match the artifacts' own prediction dicts exactly, or this
    # file has introduced the very drift it exists to prevent
    for tag, art in discover_studies():
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
    # 🔒 the guard the hand-maintained version could not have: a preregistration with no
    # scored artifact means a study ran and its predictions never reached the record, which
    # is exactly how study AQ went missing.
    # 🔒 The guard the hand-maintained version could not have: a preregistration that declares
    # predictions and that NO scored artifact points back at means a study's predictions never
    # reached the record -- exactly how study AQ went missing. Unscored ones must be named, so
    # an omission is visible instead of silent.
    declared_set = {e["prereg"] for e in exp if e["exists"]}
    unscored = sorted(declared_set - scored_preregs())
    if unscored != sorted(UNRUN):
        unexpected = sorted(set(unscored) - set(UNRUN))
        vanished = sorted(set(UNRUN) - set(unscored))
        if unexpected:
            print(f"  XX preregistrations declaring predictions with no scored artifact and "
                  f"no stated reason: {unexpected} -- either the study's verdicts never "
                  "reached the record, or the design is unrun and belongs in UNRUN")
        if vanished:
            print(f"  XX UNRUN names {vanished}, which now HAS a scored artifact -- remove "
                  "the entry so the allowlist cannot hide a future omission")
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
    for tag, _ in discover_studies():
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
        # 🔴 Was `len(rows)`, which counted the WITHDRAWN row as scored -- 42 against the
        # 41 verdicts that exist. Scored means a verdict was reached.
        "total_scored": sum(1 for r in rows if r["kind"] in ("derived", "reasoned")),
        "total_withdrawn": sum(1 for r in rows if r["kind"] == "withdrawn"),
        "by_study": by_study,
        "exposure": {
            "preregistrations": len(exp),
            "preregs_found": [e["prereg"] for e in exp],
            "studies_found": [t for t, _ in discover_studies()],
            "preregs_with_no_scored_artifact": sorted(
                {e["prereg"] for e in exp if e["exists"]} - scored_preregs()),
            "unscored_accounted_for": UNRUN,
            "declared_predictions": sum(e.get("declared_predictions", 0) for e in exp),
            # ⚠️ DECLARED and SCORED do not reconcile by subtraction, and the reason is
            # structural: DEEPVIC_BENCHMARK is scored by TWO artifacts (AI-1's metadata
            # baseline and AI's verdicts) whose P-numbers overlap, so it contributes 7
            # scored rows against 6 declarations. Reported as two separate counts rather
            # than as a balance that would have to be fudged to close.
            # 🔒 The gap between DECLARED and SCORED, reconciled from its parts instead of
            # asserted as a literal. `src/22` hard-coded `declared - scored == 6`, which went
            # stale the moment a preregistration was frozen -- the SEVENTH hand-maintained
            # number in this project to drift, inside the claim built to catch the sixth.
            # The identity: every declared prediction is scored, unrun, or withdrawn, less the
            # rows a preregistration scored twice through two artifacts.
            "reconciliation": reconcile(exp, rows),
            "preregs_scored_by_multiple_artifacts": sorted(
                {pr for pr in scored_preregs()
                 if sum(1 for _, a in discover_studies()
                        if (read(a) or {}).get("preregistration", "").endswith(pr + "_PREREGISTRATION.md")) > 1}),
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
    print(f"  scored {len(rows)} predictions across "
          f"{len({r['study'] for r in rows})} studies")
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
