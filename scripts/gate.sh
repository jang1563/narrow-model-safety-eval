#!/bin/bash
# The three-way gate, by CAPTURED exit code.
#
# 🔒 Each check's status is captured into its own variable and reported together, because the
# failure mode this guards against is a check whose exit code nobody reads. `set -e` is
# deliberately NOT used: it would abort before the later checks ran, and "the audit failed so
# we never learned whether the tests pass" is a worse report than three numbers.
#
# 🔴 This script lived in /tmp for two sessions and was lost twice when /tmp was cleared,
# which meant recreating it from memory each time -- a gate you have to remember how to run is
# a gate that gets run differently. Tracked now.
#
# Usage: bash scripts/gate.sh [outdir]   (default: .gate/)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-.gate}"
mkdir -p "$OUT"

# 🔴 Two concurrent runs wrote the same files and the report MIXED them: rc.txt said
# `pytest=0` while pytest.txt ended in a FAILED line, because each came from a different
# run. A gate whose report can be assembled from two runs is worse than no gate.
# 🔒 `mkdir` is atomic, so this is a lock and not a check-then-act.
LOCK="$OUT/.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another gate run holds $LOCK -- refusing to interleave reports" >&2
  echo "  (remove it if you are sure no gate is running)" >&2
  exit 2
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

python3 -W ignore src/22_claims_audit.py > "$OUT/audit.txt" 2>&1; A=$?
python3 -m pytest -q tests                > "$OUT/pytest.txt" 2>&1; B=$?
python3 -m ruff check src tests           > "$OUT/ruff.txt" 2>&1; C=$?

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$"
printf 'audit=%d pytest=%d ruff=%d  [%s]\n' "$A" "$B" "$C" "$STAMP" | tee "$OUT/rc.txt"
tail -1 "$OUT/audit.txt"
# ⚠️ pytest's LAST line is the summary only when it finished. An interrupted run ends in a row
# of dots with no summary and exits non-zero, which reads exactly like a passing run unless
# the exit code is checked -- that happened on 2026-10-07 and cost a wrong diagnosis.
if [ "$B" -ne 0 ]; then grep -E "^FAILED|^ERROR|assert" "$OUT/pytest.txt" | head -5; else tail -1 "$OUT/pytest.txt"; fi
[ "$C" -ne 0 ] && tail -5 "$OUT/ruff.txt"

[ "$A" -eq 0 ] && [ "$B" -eq 0 ] && [ "$C" -eq 0 ]
