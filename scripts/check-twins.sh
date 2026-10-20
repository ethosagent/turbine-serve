#!/usr/bin/env bash
# CLAUDE.md and AGENTS.md must be byte-identical so every agent runtime reads the same map.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! diff -q CLAUDE.md AGENTS.md >/dev/null; then
  echo "✗ CLAUDE.md and AGENTS.md have diverged. Fix: cp CLAUDE.md AGENTS.md"
  exit 1
fi
