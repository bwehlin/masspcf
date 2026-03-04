#!/usr/bin/env python3
"""
Generates index.html for the gh-pages coverage history site.

Usage:
    python generate_coverage_index.py <gh-pages-dir> [branch]

The script scans <gh-pages-dir>/reports/ for subdirectories named
YYYY-MM-DD_HH-MM-SS_<sha>, keeps the 5 most recent, and writes
<gh-pages-dir>/index.html.

Each report directory is expected to contain:
    cpp/coverage.html   — gcovr HTML report
    python/index.html   — pytest-cov HTML report

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


def get_entries(reports_root: str, current_branch: str | None = None) -> list[dict]:
    entries = []
    if not os.path.isdir(reports_root):
        return entries

    for i, name in enumerate(sorted(os.listdir(reports_root), reverse=True)):
        if not os.path.isdir(os.path.join(reports_root, name)):
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
        # Only the newest entry (i == 0) gets the current branch attached
        branch = current_branch if i == 0 else None
        entries.append({
            "name": name,
            "label": label,
            "sha": sha,
            "branch": branch,
            "cpp_path": f"reports/{name}/cpp/coverage.html",
            "python_path": f"reports/{name}/python/index.html",
        })
    return entries


def render_cards(entries: list[dict]) -> str:
    if not entries:
        return '<div class="empty">No coverage reports found yet.</div>'

    cards = []
    for i, entry in enumerate(entries):
        is_latest = i == 0
        badge = '<span class="badge">Latest</span>' if is_latest else ""
        branch_tag = f'<div class="report-branch">⎇ {entry["branch"]}</div>' if entry.get("branch") else ""
        cards.append(f"""
    <div class="report-card{' latest' if is_latest else ''}">
      <span class="report-index">#{i + 1}</span>
      <div class="report-info">
        <div class="report-label">{entry['label']}</div>
        {branch_tag}
      </div>
      {badge}
      <div class="report-links">
        <a class="report-link" href="{entry['cpp_path']}">C++</a>
        <a class="report-link" href="{entry['python_path']}">Python</a>
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
