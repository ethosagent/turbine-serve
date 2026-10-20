#!/usr/bin/env bash
# SessionStart (compact) hook: re-anchor the session on the live plan and any hand-off note.
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
echo "spark-engine: context was compacted. Re-read the active plan before continuing."
ls -1t plan/*.md 2>/dev/null | head -3 | sed 's/^/  plan: /'
[ -f HANDOFF.md ] && { echo "  HANDOFF.md:"; head -20 HANDOFF.md | sed 's/^/    /'; }
[ -f .logs/summary.txt ] && echo "  last check: $(cat .logs/summary.txt)"
exit 0
