"""Every `--selftest` in `src/` actually runs, and the gate runs it.

🔴 Until 2026-09-30 the pytest leg of this project's three-way gate exercised **five of 157** source
modules, all of them early-numbered. Thirty-six scripts carry a `--selftest` flag -- every study from
`src/84` onward and several before it -- and **none was run by anything**. They all passed when run by
hand, so this closes a rot path rather than fixing a break.

🔒 The list is discovered, not written down, so a new study with a selftest is covered the day it lands.
The floor below is the one hard-coded number, and it can only be raised.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# 🔒 Discovered by scanning for the flag, so this cannot fall out of date the way a hand-kept list does.
SCRIPTS = sorted(
    f for f in (ROOT / "src").glob("*.py")
    if '"--selftest"' in f.read_text(errors="replace")
    or "'--selftest'" in f.read_text(errors="replace")
)
# ⚠️ A floor, not a count: it fails if selftests are removed, and is raised deliberately when added.
SELFTEST_FLOOR = 36


def test_every_study_script_still_carries_a_selftest():
    assert len(SCRIPTS) >= SELFTEST_FLOOR, (
        f"{len(SCRIPTS)} scripts carry --selftest, below the floor of {SELFTEST_FLOOR}; "
        "a selftest was removed, or the flag was renamed"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_selftest_passes(script):
    """🔒 Runs the real thing in a subprocess, because that is how the gate would run it.

    ⚠️ A timeout rather than none: a selftest that hangs is a failure, not a slow pass.
    """
    r = subprocess.run(
        [sys.executable, str(script), "--selftest"],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )
    assert r.returncode == 0, (
        f"{script.name} --selftest exited {r.returncode}\n"
        f"{(r.stderr or r.stdout).strip()[-800:]}"
    )
    # 🔴 First written as `"PASS" in ...`, which failed `55_tier2_mutagenesis_set.py` -- it prints
    # "selftest: 18/18 pass" in lower case. The convention is not uniform across 36 scripts written
    # over five months, so the check is on the word and not on its casing.
    assert re.search(r"\bpass(?:ed)?\b", r.stdout + r.stderr, re.I), (
        f"{script.name} --selftest exited 0 without announcing a pass; "
        "a selftest that cannot announce success cannot be trusted to have run"
    )
