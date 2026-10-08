#!/usr/bin/env python3
"""Can every verdict a preregistration declares actually happen?

🔴 Three studies in a row carried a band that could not do what it was written to do, and all
three share one root cause: a band stated as an absolute COUNT against a denominator that a
LATER computation determines -- a precondition, an abstention, or the grid's own structure.

  AT-1b  "at least 5 of 6 cells clear AND no modality dead on both its encoders".
         A dead modality is two non-clearing cells, leaving at most four clears, which is
         already inside the modality-dependent band. The second clause could never bind.
  AU-3   "mean recovery below 90% AND sd above 10pp in EVERY cell-setting".
         One cell 2.6 points over the ceiling vetoed a study in which every sd had doubled.
  AV-1   "at least 6 of 8 sealed predictions correct", with an abstention region that can
         shrink the scorable pool. Two abstentions cap the attainable count at four, so a
         PERFECT record lands in the failure band of "at most 4 of 8".

🔒 Each was found after freezing, twice after results existed. All three are decidable from
the band and the unit count ALONE, before any data: that is what this file checks.

⚠️ It does not check whether a threshold is well chosen. It checks reachability -- that the
pass region is attainable, that a perfect outcome cannot score as a failure, and that the
regions do not overlap or leave an unreachable gap.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/band_reachability.json"


def check_count_band(name, n_units, pass_at_least, fail_at_most, max_unscorable=0,
                     per_unit_veto=False):
    """A band of the form 'pass if >= P of N, fail if <= F of N'.

    `max_unscorable` is how many units a precondition or abstention can remove. A unit that
    leaves the pool reduces the attainable maximum but NOT the stated denominator, which is
    exactly the defect.
    `per_unit_veto` marks a band stated as a conjunction over every unit, where one unit
    decides the study-level reading.
    """
    attainable = n_units - max_unscorable
    problems = []
    if pass_at_least > attainable:
        problems.append(
            f"PASS UNREACHABLE: needs {pass_at_least} of {n_units} but at most "
            f"{attainable} are scorable once {max_unscorable} leave the pool")
    if attainable <= fail_at_most:
        problems.append(
            f"PERFECT SCORES AS FAILURE: a flawless {attainable}/{attainable} falls inside "
            f"the failure region (<= {fail_at_most})")
    if fail_at_most >= pass_at_least:
        problems.append(
            f"REGIONS OVERLAP: <= {fail_at_most} and >= {pass_at_least} both match "
            f"{list(range(pass_at_least, fail_at_most + 1))}")
    gap = list(range(fail_at_most + 1, pass_at_least))
    if per_unit_veto and n_units > 1:
        problems.append(
            f"SINGLE-UNIT VETO: stated as a conjunction over all {n_units} units, so one "
            "marginal unit decides the study-level verdict")
    return {"band": name, "n_units": n_units, "pass_at_least": pass_at_least,
            "fail_at_most": fail_at_most, "max_unscorable": max_unscorable,
            "attainable_max": attainable, "between_bands_region": gap,
            "per_unit_veto": bool(per_unit_veto),
            "problems": problems, "ok": not problems}


def check_resolution(name, pass_at, fail_at, ci_width, n_units=None):
    """A band on a CONTINUOUS difference, against the measurement's own uncertainty.

    🔴 Study AW banded AW-2 at +1pp and AW-3 at +3pp. The paired bootstrap over its 14
    categories returns intervals **4 to 9 points wide**, so the bands were 3-9x finer than
    the measurement could resolve -- and the two arms landed on opposite sides of the +1pp
    line while BOTH intervals spanned zero. The "ARMS DISAGREE" verdict is an artifact of
    reading a point estimate against a band narrower than its own interval.

    ⚠️ `check_count_band` cannot see this: it is decidable from the band and the unit count
    alone, and this is not. It needs a variance estimate -- which study AP's per-category
    seed spreads would have supplied before AW was frozen.
    """
    problems = []
    gap = pass_at - fail_at
    if ci_width is not None and ci_width > 0:
        if pass_at <= ci_width:
            problems.append(
                f"PASS INSIDE THE NOISE: the pass threshold {pass_at:+.4g} is within a "
                f"{ci_width:.4g}-wide interval, so a null result can clear it")
        if gap < ci_width / 2:
            problems.append(
                f"BANDS FINER THAN THE MEASUREMENT: the between-bands gap is {gap:.4g} "
                f"against a {ci_width:.4g}-wide interval, so which band a point lands in "
                "is decided by noise")
    return {"band": name, "pass_at": pass_at, "fail_at": fail_at,
            "between_bands_gap": gap, "ci_width": ci_width, "n_units": n_units,
            "problems": problems, "ok": not problems}


def check_implied(name, outer, inner_description, implied):
    """A conjunction between a count band and a structural clause the count already implies."""
    return {"band": name, "outer": outer, "inner": inner_description,
            "problems": ([f"CLAUSE IS IMPLIED: {inner_description} forces the count into "
                          f"{outer}, so the conjunction adds nothing"] if implied else []),
            "ok": not implied}


# 🔒 The three known cases, as a regression set. If a future change to this file stops
# flagging any of them, the file has stopped working.
KNOWN = [
    lambda: check_count_band("AV-1 (TRAINING_FREE_RULE)", n_units=8, pass_at_least=6,
                             fail_at_most=4, max_unscorable=4),
    lambda: check_count_band("AU-3 (IMAGE_AXIS_REPAIR)", n_units=4, pass_at_least=4,
                             fail_at_most=0, max_unscorable=0, per_unit_veto=True),
    # 🔴 Study AW: bands at 1pp and 3pp against bootstrap intervals 4-9pp wide.
    lambda: check_resolution("AW-2 (MARGIN_ROUTING)", pass_at=0.03, fail_at=0.01,
                             ci_width=0.069, n_units=14),
    lambda: check_implied("AT-1b (GENERALITY_GRID)",
                          outer="the 3-4 of 6 modality-dependent band",
                          inner_description="a modality dead on both its encoders",
                          implied=True),
]


def selftest():
    ok = True
    # 🔒 all three known defects must be flagged
    for fn in KNOWN:
        r = fn()
        if r["ok"]:
            print(f"  XX {r['band']} was not flagged")
            ok = False
    aw2 = [fn() for fn in KNOWN if fn().get("band", "").startswith("AW-2")][0]
    if not any("BANDS FINER THAN THE MEASUREMENT" in p for p in aw2["problems"]):
        print(f"  XX AW-2's resolution problem was not named: {aw2['problems']}")
        ok = False
    if not any("PASS INSIDE THE NOISE" in p for p in aw2["problems"]):
        print(f"  XX AW-2's pass-inside-the-noise was not named: {aw2['problems']}")
        ok = False
    # 🔒 a band comfortably wider than its interval must come back clean
    wide = check_resolution("wide", pass_at=0.20, fail_at=0.05, ci_width=0.02)
    if not wide["ok"]:
        print(f"  XX a well-resolved band was flagged: {wide['problems']}")
        ok = False
    # 🔒 and with no variance estimate the check must stay silent rather than guess
    if not check_resolution("unknown", 0.03, 0.01, None)["ok"]:
        print("  XX a band with no interval estimate was flagged anyway")
        ok = False

    av1 = KNOWN[0]()
    if not any("PERFECT SCORES AS FAILURE" in p for p in av1["problems"]):
        print(f"  XX AV-1's perfect-score-as-failure was not named: {av1['problems']}")
        ok = False
    if not any("PASS UNREACHABLE" in p for p in av1["problems"]):
        print(f"  XX AV-1's unreachable pass was not named: {av1['problems']}")
        ok = False
    # 🔒 and a well-formed band must come back clean, or the check calls everything broken
    good = check_count_band("well formed", n_units=10, pass_at_least=7, fail_at_most=3,
                            max_unscorable=2)
    if not good["ok"]:
        print(f"  XX a well-formed band was flagged: {good['problems']}")
        ok = False
    if good["between_bands_region"] != [4, 5, 6]:
        print(f"  XX the between-bands region is wrong: {good['between_bands_region']}")
        ok = False
    # overlap must be caught
    if check_count_band("overlapping", 10, 4, 5)["ok"]:
        print("  XX overlapping regions were not caught")
        ok = False
    # a band whose pass needs every single unit is reachable but fragile, not an error
    tight = check_count_band("tight", n_units=4, pass_at_least=4, fail_at_most=0)
    if not tight["ok"]:
        print(f"  XX a tight-but-reachable band was flagged: {tight['problems']}")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        epilog="With no arguments, reports the three known cases as a regression set.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--units", type=int)
    ap.add_argument("--pass-at-least", type=int)
    ap.add_argument("--fail-at-most", type=int)
    ap.add_argument("--max-unscorable", type=int, default=0)
    ap.add_argument("--veto", action="store_true",
                    help="the band is a conjunction over every unit")
    ap.add_argument("--name", default="band")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if a.units is not None:
        if a.pass_at_least is None or a.fail_at_most is None:
            raise SystemExit("--units needs --pass-at-least and --fail-at-most")
        rows = [check_count_band(a.name, a.units, a.pass_at_least, a.fail_at_most,
                                 a.max_unscorable, a.veto)]
    else:
        rows = [fn() for fn in KNOWN]

    for r in rows:
        print(f"  {r['band']}")
        for k in ("n_units", "attainable_max", "between_bands_region"):
            if k in r:
                print(f"      {k}: {r[k]}")
        for p in r["problems"]:
            print(f"      🔴 {p}")
        if r["ok"]:
            print("      ok: every region is reachable")
    bad = [r for r in rows if not r["ok"]]
    if a.units is not None:
        # 🔴 An ad-hoc query used to overwrite the regression artifact, so checking a proposed
        # band replaced the three known cases that claim 154 audits -- caught by the gate
        # immediately, but a query must not be able to clobber a fixture.
        print(f"\n{len(bad)} of {len(rows)} band(s) flagged "
              f"(ad-hoc query; {OUT.name} not written)")
        return 0
    OUT.write_text(json.dumps(
        {"study": "band reachability -- can every declared verdict happen?",
         "why": ("three studies in a row carried a band that could not do what it was "
                 "written to do, all decidable from the band and the unit count alone"),
         "rows": rows, "n_flagged": len(bad)}, indent=2) + "\n")
    print(f"\n{len(bad)} of {len(rows)} band(s) flagged")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
