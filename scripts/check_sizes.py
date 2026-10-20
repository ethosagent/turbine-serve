"""Gate: file-length limits so one read captures one unit of meaning. Prints violations only.

Rows are (glob, warn_above, fail_above). docs/SPEC.md is exempt (the input spec).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES: list[tuple[str, int | None, int | None]] = [
    ("spark_engine/**/*.py", 300, 800),
    ("tests/**/*.py", 500, None),
    ("docs/*.md", None, 200),
    ("plan/decisions/*.md", None, 200),
    ("docs/runbooks/*.md", None, 200),
    ("CLAUDE.md", None, 150),
    ("AGENTS.md", None, 150),
    ("ARCHITECTURE.md", None, 300),
    (".claude/skills/*/SKILL.md", None, 300),
]
EXEMPT = {ROOT / "docs" / "SPEC.md"}


def main() -> int:
    failed = False
    for pattern, warn, fail in RULES:
        for path in sorted(ROOT.glob(pattern)):
            if path in EXEMPT:
                continue
            n = sum(1 for _ in path.open(encoding="utf-8"))
            rel = path.relative_to(ROOT)
            if fail is not None and n > fail:
                print(f"FAIL: {rel}: {n} lines > {fail} (hard limit)")
                failed = True
            elif warn is not None and n > warn:
                print(f"warn: {rel}: {n} lines > {warn} (soft limit)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
