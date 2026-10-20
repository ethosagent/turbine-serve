#!/usr/bin/env bash
# The single check set. `make check`, the pre-push hook and CI all run THESE scripts/check-*.sh,
# so local == hook == CI by construction. Runs every gate, reports ALL failures, exit 1 if any.
# Output is failures only; every gate's full log is in .logs/<gate>.log. CHECK_VERBOSE=1 prints all.
# Add a gate: drop scripts/check-<name>.sh here; it is discovered automatically.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GATES=(); for f in scripts/check-*.sh; do f=${f#scripts/check-}; GATES+=("${f%.sh}"); done
mkdir -p .logs
failures=(); started=$(date +%s)
for name in "${GATES[@]}"; do
  s=$(date +%s)
  if bash "scripts/check-$name.sh" >".logs/$name.log" 2>&1; then
    echo "✓ $name ($(( $(date +%s) - s ))s)"
    [ "${CHECK_VERBOSE:-}" = "1" ] && cat ".logs/$name.log"
  else
    failures+=("$name")
    echo "✗ $name FAILED ($(( $(date +%s) - s ))s)  → .logs/$name.log"
    tail -n 40 ".logs/$name.log" | sed 's/^/    /'
  fi
done
echo "=== summary === ($(( $(date +%s) - started ))s)"
if [ ${#failures[@]} -ne 0 ]; then
  echo "✗ failed: ${failures[*]}" | tee .logs/summary.txt; exit 1
fi
echo "✓ all checks passed" | tee .logs/summary.txt
