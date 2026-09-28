"""Fail when a Python file exceeds the line limit (pre-commit hook).

Ruff has no file-length rule, so this enforces the project's target of at most
500 lines per module. The fix for a failure is to split the module, not to raise
the limit.

Usage:
    python scripts/check_file_length.py [--max N] FILE [FILE ...]
"""

import argparse
import sys
from pathlib import Path

DEFAULT_MAX_LINES = 500


def main(argv: list[str] | None = None) -> int:
    """Check each file's line count and report any over the limit.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 if every file is within the limit, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max", type=int, default=DEFAULT_MAX_LINES, help="maximum lines per file")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    too_long = []
    for path in args.files:
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > args.max:
            too_long.append((path, lines))

    for path, lines in too_long:
        print(f"{path}: {lines} lines (max {args.max}); split this module")
    return 1 if too_long else 0


if __name__ == "__main__":
    sys.exit(main())
