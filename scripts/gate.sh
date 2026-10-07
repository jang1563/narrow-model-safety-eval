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

python3 -W ignore src/22_claims_audit.py > "$OUT/audit.txt" 2>&1; A=$?
python3 -m pytest -q tests                > "$OUT/pytest.txt" 2>&1; B=$?
python3 -m ruff check src tests           > "$OUT/ruff.txt" 2>&1; C=$?

printf 'audit=%d pytest=%d ruff=%d\n' "$A" "$B" "$C" | tee "$OUT/rc.txt"
tail -1 "$OUT/audit.txt"
# ⚠️ pytest's LAST line is the summary only when it finished. An interrupted run ends in a row
# of dots with no summary and exits non-zero, which reads exactly like a passing run unless
# the exit code is checked -- that happened on 2026-10-07 and cost a wrong diagnosis.
if [ "$B" -ne 0 ]; then grep -E "^FAILED|^ERROR|assert" "$OUT/pytest.txt" | head -5; else tail -1 "$OUT/pytest.txt"; fi
[ "$C" -ne 0 ] && tail -5 "$OUT/ruff.txt"

[ "$A" -eq 0 ] && [ "$B" -eq 0 ] && [ "$C" -eq 0 ]
