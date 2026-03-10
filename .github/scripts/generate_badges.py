#!/usr/bin/env python3
"""
Generates shields.io endpoint JSON badge files for C++ and Python coverage.

Usage:
    python generate_badges.py <gh-pages-dir> <covreport-dir>
"""

import json
import sys
from pathlib import Path


def coverage_color(pct: float) -> str:
    if pct >= 80:
        return "brightgreen"
    elif pct >= 60:
        return "yellow"
    else:
        return "red"


def write_badge(path: Path, label: str, pct: float) -> None:
    badge = {
        "schemaVersion": 1,
        "label": label,
        "message": f"{pct:.1f}%",
        "color": coverage_color(pct),
    }
    path.write_text(json.dumps(badge))
    print(f"Written {path} ({pct:.1f}%)")


def read_cpp_coverage(covreport_dir: Path) -> float:
    summary = covreport_dir / "cpp" / "coverage-summary.json"
    with summary.open() as f:
        return json.load(f)["line_percent"]


def read_python_coverage(covreport_dir: Path) -> float:
    summary = covreport_dir / "python" / "coverage.json"
    with summary.open() as f:
        return json.load(f)["totals"]["percent_covered"]


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <gh-pages-dir> <covreport-dir>", file=sys.stderr)
        sys.exit(1)

    gh_pages_dir = Path(sys.argv[1])
    covreport_dir = Path(sys.argv[2])

    cpp_pct = read_cpp_coverage(covreport_dir)
    py_pct = read_python_coverage(covreport_dir)

    write_badge(gh_pages_dir / "cpp-coverage-badge.json", "C++ coverage", cpp_pct)
    write_badge(gh_pages_dir / "py-coverage-badge.json", "Python coverage", py_pct)


if __name__ == "__main__":
    main()
