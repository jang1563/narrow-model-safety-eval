#!/usr/bin/env python3
"""
98_manuscript_figure_audit.py - which numbers in the manuscript appear in NO artifact at all?

🔴 READ THIS BEFORE TRUSTING A PASS. This began as a document-wide audit -- "does every figure trace to
an artifact?" -- and its own null calibration showed it cannot do that job. Perturbing a matched token
so it must round differently and re-testing gives a false-pass rate of 94% at three significant digits
and 48% at four. The cause is structural: 16,748 of the 19,171 artifact values lie in [0, 1), so any
two-to-four-significant-figure rate finds a neighbour by coincidence. Tightening does not fix it,
because the manuscript quotes two to four significant figures.

🔒 So a MATCH here means almost nothing and is never evidence that a figure is current. src/22's pins
are the only mechanism in this repository that can establish that, because they attach a figure to the
artifact it actually came from. What this tool can still do is the narrower job in its title: list
figures that appear NOWHERE, which catches a number invented from nothing rather than one merely out
of date.

`src/22` pins individual SENTENCES to claims, which catches a pinned figure that drifts. It cannot
catch a figure nobody pinned, and the manuscript has been patched more than a dozen times since entry
52 -- the VFDB reference moved, five share-of-gap figures moved, H-5 was withdrawn as a share, the
length-U section was written and then rewritten twice. A stale number in an unpinned sentence would
survive all of that.

This extracts every numeric token in the manuscript and asks whether some results artifact holds a
value that ROUNDS to it. Unmatched tokens are candidates for staleness, not verdicts: years, citation
figures, other papers' numbers and frozen thresholds stated in prose are legitimately absent from this
repository's artifacts, and those live in ALLOW below with a reason each.

The null calibration runs on every invocation and is printed first, so this can never quietly become
something it is not.

Usage:
    python src/98_manuscript_figure_audit.py
    python src/98_manuscript_figure_audit.py --selftest
"""

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "paper" / "MANUSCRIPT.md"
OUT = ROOT / "results" / "manuscript_figure_audit.json"
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+)(?![\w])")

# 🔒 Numbers that legitimately have no artifact in this repository. Each needs a reason, because an
# allowlist without reasons is a way to make a failing check pass.
ALLOW = {
    "0.05": "the nominal false-positive budget, a design constant",
    "0.95": "its complement, the calibration quantile",
    "5.0": "the nominal 5% written as a bare number in prose",
    "2.0": "a frozen band boundary (matched-VFDB C/A, and the U-index bands)",
    "3.0": "a frozen band boundary (the localization R bands)",
    "1.5": "a frozen band boundary (the localization R lower band)",
    "0.65": "the frozen length-confound AUROC tolerance",
    "0.5": "the midpoint an AUROC is compared against",
    "1.0": "the null value a ratio is compared against",
    "0.92": "DTVF's reported AUROC, from the published paper (entry 33)",
    "0.754": "the amino-acid-composition baseline, reported in section 2 from the v2 panel",
    "0.506": "the shuffled-label baseline, same source",
    "0.818": "the provenance-probe AUROC, same source",
    "0.981": "the SUPERSEDED v1 separability, named in the text as superseded",
}


MAX_LIST = 5


def artifact_values(paths):
    """Numeric leaves in every artifact, EXCLUDING long lists.

    🔴 The first version took every numeric leaf and reached 47,833 values, at which point a null
    calibration showed the audit passed a deliberately wrong number 93.5% of the time -- with that many
    values in the space, almost any token finds a match by coincidence. A check that cannot fail is
    worse than no check, because it manufactures confidence.

    🔒 The flood comes from per-seed arrays, bootstrap draws and per-protein lists, which a manuscript
    never quotes. Lists longer than MAX_LIST are skipped, leaving the scalars and short tuples --
    means, intervals, counts -- that a write-up actually cites. The null calibration is reported on
    every run so this can never silently become vacuous again.
    """
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            if len(o) > MAX_LIST:
                return
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            vals.add(float(o))

    for p in paths:
        try:
            walk(json.loads(p.read_text()))
        except Exception:                                              # noqa: BLE001
            continue
    return vals


def matches(token, vals):
    """Does some artifact value round to this token, as a number or as a percentage?"""
    raw = token.replace(",", "")
    try:
        x = float(raw)
    except ValueError:
        return False
    dec = len(raw.split(".")[1]) if "." in raw else 0
    tol = 0.5 * 10 ** (-dec)
    for cand in (x, x / 100.0):
        t = tol if cand == x else tol / 100.0
        for v in vals:
            if abs(v - cand) <= t:
                return True
    return False


def selftest():
    vals = {0.21747, 4218.0, 0.9193}
    assert matches("21.75", vals), "a percentage must match its fraction in the artifact"
    assert matches("4,218", vals), "comma thousands must match"
    assert matches("91.93", vals)
    assert not matches("21.80", vals), "a value that does not round to the token must not match"
    assert not matches("999.9", vals)
    # 🔒 tolerance must scale with the decimals shown, not be fixed. 0.24 DOES round to 0.2 at one
    # decimal and must match; 0.26 rounds to 0.3 and must not. The first version of this assertion had
    # it backwards and the selftest caught it.
    assert matches("0.2", {0.24}) is True, "0.24 rounds to 0.2 at one decimal"
    assert matches("0.2", {0.26}) is False, "0.26 rounds to 0.3, not 0.2"
    assert matches("0.24", {0.24}) is True and matches("0.24", {0.21}) is False
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return selftest()

    paths = sorted(ROOT.glob("results/*.json")) + sorted(ROOT.glob("results/v3/*.json")) \
        + sorted(ROOT.glob("results/v2/*.json"))
    vals = artifact_values(paths)
    text = DOC.read_text()
    lines = text.splitlines()
    seen, unmatched = {}, []
    for i, line in enumerate(lines, 1):
        for tok in NUM.findall(line):
            seen.setdefault(tok, []).append(i)
    for tok, where in sorted(seen.items()):
        if tok in ALLOW or matches(tok, vals):
            continue
        unmatched.append({"token": tok, "lines": where[:4], "n_occurrences": len(where)})

    # ---- 🔴 is this check capable of failing? ----------------------------------------------------
    # With 47k artifact values a token could match by coincidence, and a check that almost everything
    # passes is not evidence until its false-pass rate is known. Every matched token is perturbed --
    # its last significant digit shifted by enough to change the rounding -- and re-tested. A high
    # pass rate on perturbed values would mean this audit cannot detect a stale figure.
    import random
    rng = random.Random(0)
    tested = [t for t in seen if t not in ALLOW and matches(t, vals)]
    false_pass = 0
    for tok in tested:
        raw = tok.replace(",", "")
        dec = len(raw.split(".")[1]) if "." in raw else 0
        step = 10 ** (-dec)
        bogus = float(raw) + step * rng.choice([-3, -2, 2, 3])
        fmt = f"{bogus:.{dec}f}"
        if matches(fmt, vals):
            false_pass += 1
    fp_rate = false_pass / len(tested) if tested else float("nan")
    print(f"\n🔴 null calibration: {false_pass} of {len(tested)} PERTURBED tokens also match "
          f"({fp_rate * 100:.1f}%) — the rate at which this audit passes a wrong number")

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "null_calibration": {"n_tested": len(tested), "n_false_pass": false_pass,
                                "false_pass_rate": fp_rate},
           "n_artifacts": len(paths), "n_artifact_values": len(vals),
           "n_distinct_tokens": len(seen), "n_allowlisted": sum(1 for t in seen if t in ALLOW),
           "n_unmatched": len(unmatched), "unmatched": unmatched}
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{len(paths)} artifacts -> {len(vals)} distinct numeric values")
    print(f"{len(seen)} distinct numeric tokens in {DOC.relative_to(ROOT)}, "
          f"{res['n_allowlisted']} allowlisted")
    print(f"\n{len(unmatched)} appear in no artifact — the only informative output here:\n")
    for u in unmatched:
        ctx = lines[u["lines"][0] - 1].strip()
        print(f"  {u['token']:>10}  line {u['lines'][0]:<4} x{u['n_occurrences']}  {ctx[:88]}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
