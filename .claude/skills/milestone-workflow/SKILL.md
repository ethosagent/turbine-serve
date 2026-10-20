---
name: milestone-workflow
description: >-
  The workflow spine for changing turbine-serve code: pick a plan task, write the failing test,
  implement the block, run make check, commit. Use for ANY feature, fix, refactor or plan task in
  this repo ("implement task 4", "add the scheduler", "fix the sampler", "continue the plan").
---

# milestone-workflow

Rules that must never be missed:

- **Work from a plan.** A task plan names the task, its files, interfaces and tests. Read the
  task's section, not the whole plan. No plan section for the change? Write one first (10 lines
  is fine) so the next session can pick it up.
- **Test first.** Write the test named in the plan, run it, watch it fail for the right reason,
  then implement. `uv run pytest tests/test_<block>.py -q`.
- **One block, one file, one component.** New behaviour goes in the file that owns it
  (`CLAUDE.md` "Where new code goes"). `get_logger("<file stem>")` only.
- **Contracts first.** If the change needs a new field, add it to `types.py` with a default,
  then update every consumer; if it needs a new seam, add the Protocol to `interfaces.py`.
  A `types.py` change needs a test in each consumer it touches.
- **D1–D10 are not yours to change.** If the task needs it, stop and ask the owner.
- **Done = `make check` green + tests for the change + docs/skills updated + one commit.**
  Never say it passes without re-running it. Gotcha discovered? Append it to
  `.claude/skills/engine-debug/SKILL.md` in the same commit.

## Procedure

1. **Orient.** `make check` to confirm the tree is green before you start. Read the plan task's
   Files, Interfaces and Steps. Read `docs/SPEC.md` for the block (its numbered section only).
2. **Place.** Confirm the file from the plan matches the dependency law (`ARCHITECTURE.md` §II).
   A new module that imports a sibling block is wrong; it must consume a Protocol instead.
3. **Red.** Write the failing test(s) exactly as the plan names them, with the plan's values.
   Run the one file. Expected: fail with ImportError or AssertionError, not a typo.
4. **Green.** Implement the minimal code. Keep the signatures from the plan's Interfaces block
   verbatim; later tasks depend on those names.
5. **Shape comments.** Every tensor-producing line gets `# [T, H, D]`-style shape notes.
6. **Check.** `make check`. Fix every failing gate; `.logs/<gate>.log` has the detail. Data-plane
   blocks are tested on CPU against the tiny checkpoint (`tiny_model_dir` fixture); only the
   real-model verification runs on the Spark, and a Spark failure is reproduced on the Mac before it is fixed.
7. **Docs in the same commit.** New config flag → `README.md` run section if user-facing. New
   event name → nothing (logs are self-describing) unless it changes the jq trail in
   `docs/orientation.md`. New gotcha → `engine-debug` skill.
8. **Commit.** One concern, imperative subject (`feat: FCFS scheduler with KV admission`),
   attribution line from the session. Tick the task's checkboxes in the plan.
9. **Hand off** when the task is done or context is past ~200K: `/handoff`.

## When the plan and reality disagree

- Spec value wrong (shape, tensor name, token id)? Verify against `docs/qwen3-and-spark.md` or
  `AutoConfig`, fix the plan line, note it in the commit body.
- A test in the plan cannot pass as written? Say which assertion and why before changing it.
- Interfaces need to change? Update the plan's Interfaces block first, then every task that
  consumes it, then the code.
