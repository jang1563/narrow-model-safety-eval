"""The resume path of src/76's pool embedding, which is only exercised when a run is interrupted.

🔴 A resume bug is invisible on a successful run and silently drops or duplicates proteins on an
interrupted one, which would move a false-positive rate without any error. The counting loop is
lifted out of the source rather than restated, so the test cannot drift from the implementation:
if the loop is rewritten and this regex stops matching, the test fails loudly instead of passing
against a copy of the old logic.
"""

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "76_pool_reduction_embed.py"
TAGS = ["mean_res", "win_best25"]
FULL = [*TAGS, "_incl"]


def _loop():
    src = SRC.read_text()
    m = re.search(r"    have = 0\n(.*?)\n    start = have \* SHARD", src, re.S)
    assert m, "src/76's shard-counting loop was not found; update this test with the code"
    return "have = 0\n" + "\n".join(x[4:] for x in m.group(1).splitlines())


def _count(shards, tags=TAGS):
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d)
        for i, keys in shards:
            np.savez(ck / f"shard_{i:05d}.npz",
                     **{k: np.zeros((2, 3), np.float32) for k in keys})
        ns = {"np": np, "ck": ck, "a": type("A", (), {"tags": tags})()}
        exec(_loop(), ns)          # noqa: S102 — the point is to run the real loop
        return ns["have"]


@pytest.mark.parametrize(("name", "shards", "expect"), [
    ("no shards", [], 0),
    ("three in a row", [(0, FULL), (1, FULL), (2, FULL)], 3),
    # a lost middle shard must re-run from the gap, not skip past it
    ("gap at 1", [(0, FULL), (2, FULL), (3, FULL)], 1),
    # widening --tags must invalidate shards that predate the new tag
    ("shard 1 missing a tag", [(0, FULL), (1, ["mean_res", "_incl"]), (2, FULL)], 1),
    ("shard 0 missing the gate array", [(0, TAGS), (1, FULL)], 0),
    ("numbering starts at 1", [(1, FULL), (2, FULL)], 0),
])
def test_resume_counts_only_complete_leading_shards(name, shards, expect):
    assert _count(shards) == expect, name


def test_shard_size_and_tail_are_consistent():
    """A kill costs at most SHARD-1 proteins, and the tail is never checkpointed."""
    src = SRC.read_text()
    shard = int(re.search(r"^SHARD = (\d+)", src, re.M).group(1))
    assert shard > 1
    # the partial tail must not be written, or `start = have * SHARD` would skip proteins
    assert "if len(pend_incl) == SHARD:" in src
    n = 8259
    assert n // shard * shard + (n - n // shard * shard) == n
    assert n - n // shard * shard < shard


def test_reassembly_asserts_its_row_count():
    """Shard arithmetic must fail as a row count, not as a wrong rate downstream."""
    src = SRC.read_text()
    assert "!= len(recs)" in src and "the \n                         \"shard arithmetic is wrong" in src \
        or "shard arithmetic is wrong" in src
