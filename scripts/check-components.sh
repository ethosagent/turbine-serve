#!/usr/bin/env bash
# Design rule 1 and 5: one block = one file, and the log `component` is the file name.
set -euo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -I scripts/check_components.py
