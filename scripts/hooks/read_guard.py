"""PreToolUse hook on Read: block whole-file reads over 400 lines; ask for a ranged read.

plan/ files and HANDOFF.md are the hand-off and may be read whole. Exit 2 blocks the call and
shows the stderr message to the agent.
"""

import json
import sys

LIMIT = 400


def main() -> int:
    inp = json.load(sys.stdin).get("tool_input", {})
    path = inp.get("file_path", "")
    if not path or inp.get("limit") or "/plan/" in path or path.endswith("/HANDOFF.md"):
        return 0
    try:
        with open(path, "rb") as f:
            for n, _ in enumerate(f, 1):
                if n > LIMIT:
                    break
            else:
                return 0
    except OSError:
        return 0
    print(
        f"{path} is over {LIMIT} lines. Read it in ranges (offset/limit <= 200) or grep for "
        "the symbol you need.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
