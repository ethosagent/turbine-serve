#!/usr/bin/env bash
# Write the mechanical half of HANDOFF.md (git state, last gate result, files touched).
# The /handoff skill fills in Done / Remaining / Open decisions above the generated block.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
{
  cat <<HEAD
# HANDOFF — $(date -u +%Y-%m-%dT%H:%MZ)

## Done
- (fill in)

## Remaining
- (fill in)

## Open decisions
- (fill in)

## Generated
- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo none) @ $(git rev-parse --short HEAD 2>/dev/null || echo no-commits)
- last check: $(cat .logs/summary.txt 2>/dev/null || echo 'not run')
- files touched (git status --short):
HEAD
  git status --short 2>/dev/null | sed 's/^/    /'
  echo "- diff stat:"
  git diff --stat 2>/dev/null | tail -n 15 | sed 's/^/    /'
} > HANDOFF.md
echo "wrote HANDOFF.md"
