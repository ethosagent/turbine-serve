"""Gate: every get_logger("<component>") call names the file it lives in.

A block's log component must be its module stem (scheduler.py -> "scheduler") or its package
name for package-shaped blocks (api/server.py -> "api"). Keeps `jq 'select(.component==...)'`
honest and stops one block from logging as another. Prints violations only.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "turbine_serve"


def main() -> int:
    bad: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        ok = sorted({path.stem, rel.parts[0].removesuffix(".py")})
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "get_logger"):
                continue
            arg = node.args[0] if node.args else None
            if not isinstance(arg, ast.Constant):
                bad.append(f"{path}:{node.lineno}: get_logger() must take a string literal")
            elif arg.value not in ok:
                bad.append(f"{path}:{node.lineno}: get_logger({arg.value!r}); allowed: {ok}")
    print("\n".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
