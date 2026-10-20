---
name: handoff
description: >-
  End a session cleanly: write HANDOFF.md (done, remaining, failing gate, open decisions, files
  touched) so a fresh session can continue from a short note instead of a compacted transcript.
  Use when the task is done, context is past ~200K tokens, or the user says "hand off",
  "wrap up", "save state", "/handoff".
---

# handoff

1. Run `make check` one last time so the hand-off records a true gate state.
2. `bash scripts/handoff.sh` writes `HANDOFF.md` with the generated block (branch, last check,
   active plan, `git status`, diff stat).
3. Fill the three sections above the generated block, each as short bullets:
   - **Done:** plan tasks completed this session (task numbers), commits made.
   - **Remaining:** the next task number and its first step; anything half-finished and where.
   - **Open decisions:** questions waiting on the owner, with the options you see.
4. Tick the completed checkboxes in the active `plan/*.md`.
5. If the Spark was used: state whether `qwen38-flash` has been started again.
6. Tell the user: the hand-off path, one line on where to resume, and suggest `/clear`.

`HANDOFF.md` is gitignored; it is for the next session, not for history. Durable facts go in the
plan file or a skill.
