#!/usr/bin/env bash
# Size gates from the Context Budget Playbook: files an agent can read in one go.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -I scripts/check_sizes.py
