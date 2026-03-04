#!/usr/bin/env python3
"""
Generates index.html for the gh-pages coverage history site.

Usage:
    python generate_coverage_index.py <gh-pages-dir> [branch]

The script scans <gh-pages-dir>/reports/ for subdirectories named
YYYY-MM-DD_HH-MM-SS_<sha>, keeps the 5 most recent, and writes
<gh-pages-dir>/index.html.

Each report directory is expected to contain:
    cpp/coverage.html             — gcovr HTML report
    python/index.html             — pytest-cov HTML report
    valgrind/vg_pytest/index.html — ValgrindCI HTML report (pytest)
    valgrind/vg_gtest/index.html  — ValgrindCI HTML report (gtest)

Template files are resolved relative to this script's location:
    .github/ci/coverage/index.template.html
    .github/ci/coverage/style.css
"""

import os
import re
import shutil
import sys

MAX_REPORTS = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "..", "ci", "coverage")


def load_template() -> str:
    path = os.path.join(TEMPLATES_DIR, "index.template.html")
    with open(path) as f:
        return f.read()


def load_css() -> str:
    path = os.path.join(TEMPLATES_DIR, "style.css")
    with open(path) as f:
        return f.read()


def prune_old_reports(reports_root: str) -> None:
    if not os.path.isdir(reports_root):
        return
    dirs = sorted(
        [d for d in os.listdir(reports_root) if os.path.isdir(os.path.join(reports_root, d))],
        reverse=True,
    )
    for old in dirs[MAX_REPORTS:]:
        old_path = os.path.join(reports_root, old)
        print(f"Removing old report: {old_path}")
        shutil.rmtree(old_path)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def parse_cpp_coverage(report_dir: str) -> str | None:
    """Return line coverage % string from gcovr HTML, e.g. '64.6%'."""
    html = _read(os.path.join(report_dir, "cpp", "coverage.html"))
    # gcovr emits: <td class="headerCovTableEntry...">64.6 %</td>
    # The first such entry in the summary table is line coverage.
    m = re.search(r'headerCovTableEntry[^"]*">\s*([\d.]+)\s*%\s*</td>', html)
    return f"{m.group(1)}%" if m else None


def parse_python_coverage(report_dir: str) -> str | None:
    """Return total coverage % string from coverage.py HTML, e.g. '69%'."""
    html = _read(os.path.join(report_dir, "python", "index.html"))
    # coverage.py emits: <span class="pc_cov">69%</span>
    m = re.search(r'class=["\']pc_cov["\']>\s*([\d.]+%)\s*<', html)
    return m.group(1) if m else None


def parse_valgrind_errors(report_dir: str, suite: str) -> str | None:
    """Return total error count from ValgrindCI HTML, e.g. '0 errors'."""
    html = _read(os.path.join(report_dir, "valgrind", suite, "index.html"))
    # ValgrindCI summary table has a "Total" row with an error count.
    # Look for a digit sequence near the word "Total" or in the summary header.
    m = re.search(r'[Tt]otal\D{0,40}?(\d+)\s*error', html)
    if m:
        n = int(m.group(1))
        return f"{n} error{'s' if n != 1 else ''}"
    # Fallback: find any "N errors" pattern
    m = re.search(r'(\d+)\s*error', html)
    if m:
        n = int(m.group(1))
        return f"{n} error{'s' if n != 1 else ''}"
    return None


def get_entries(reports_root: str, current_branch: str | None = None) -> list[dict]:
    entries = []
    if not os.path.isdir(reports_root):
        return entries

    for i, name in enumerate(sorted(os.listdir(reports_root), reverse=True)):
        report_dir = os.path.join(reports_root, name)
        if not os.path.isdir(report_dir):
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_([0-9a-f]+)", name)
        if m:
            date_str = m.group(1)
            time_str = m.group(2).replace("-", ":")
            sha = m.group(3)
            label = f"{date_str} {time_str} UTC — {sha}"
        else:
            label = name
            sha = ""

        branch = current_branch if i == 0 else None
        entries.append({
            "name": name,
            "label": label,
            "sha": sha,
            "branch": branch,
            "cpp_coverage": parse_cpp_coverage(report_dir),
            "python_coverage": parse_python_coverage(report_dir),
            "valgrind_pytest_errors": parse_valgrind_errors(report_dir, "vg_pytest"),
            "valgrind_gtest_errors": parse_valgrind_errors(report_dir, "vg_gtest"),
            "cpp_path": f"reports/{name}/cpp/coverage.html",
            "python_path": f"reports/{name}/python/index.html",
            "valgrind_pytest_path": f"reports/{name}/valgrind/vg_pytest/index.html",
            "valgrind_gtest_path": f"reports/{name}/valgrind/vg_gtest/index.html",
        })
    return entries


def _stat_badge(value: str | None, is_error: bool = False) -> str:
    """Render a small inline stat next to a link button."""
    if value is None:
        return ""
    css_class = "stat-error" if is_error and value != "0 errors" else "stat-ok"
    return f'<span class="stat {css_class}">{value}</span>'


def render_cards(entries: list[dict]) -> str:
    if not entries:
        return '<div class="empty">No coverage reports found yet.</div>'

    cards = []
    for i, entry in enumerate(entries):
        is_latest = i == 0
        badge = '<span class="badge">Latest</span>' if is_latest else ""
        branch_tag = f'<div class="report-branch">⎇ {entry["branch"]}</div>' if entry.get("branch") else ""

        cpp_stat    = _stat_badge(entry["cpp_coverage"])
        python_stat = _stat_badge(entry["python_coverage"])
        vg_pytest_stat = _stat_badge(entry["valgrind_pytest_errors"], is_error=True)
        vg_gtest_stat  = _stat_badge(entry["valgrind_gtest_errors"],  is_error=True)

        cards.append(f"""
    <div class="report-card{' latest' if is_latest else ''}">
      <span class="report-index">#{i + 1}</span>
      <div class="report-info">
        <div class="report-label">{entry['label']}</div>
        {branch_tag}
      </div>
      {badge}
      <div class="report-links">
        <a class="report-link" href="{entry['cpp_path']}">C++ {cpp_stat}</a>
        <a class="report-link" href="{entry['python_path']}">Python {python_stat}</a>
        <a class="report-link" href="{entry['valgrind_pytest_path']}">Valgrind pytest {vg_pytest_stat}</a>
        <a class="report-link" href="{entry['valgrind_gtest_path']}">Valgrind gtest {vg_gtest_stat}</a>
      </div>
    </div>""")
    return "\n".join(cards)


def render_html(entries: list[dict]) -> str:
    template = load_template()
    css = load_css()
    cards = render_cards(entries)
    return (template
            .replace("{{style}}", css)
            .replace("{{cards}}", cards)
            .replace("{{max_reports}}", str(MAX_REPORTS)))


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gh-pages-dir> [branch]", file=sys.stderr)
        sys.exit(1)

    gh_pages_dir = sys.argv[1]
    branch = sys.argv[2] if len(sys.argv) >= 3 else None
    reports_root = os.path.join(gh_pages_dir, "reports")

    prune_old_reports(reports_root)
    entries = get_entries(reports_root, branch)
    html = render_html(entries)

    out_path = os.path.join(gh_pages_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Written {out_path} with {len(entries)} report(s).")


if __name__ == "__main__":
    main()
