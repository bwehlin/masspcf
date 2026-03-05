#!/usr/bin/env python3
"""
Generates index.html for the gh-pages coverage history site.

Usage:
    python generate_coverage_index.py <gh-pages-dir>

The script scans <gh-pages-dir>/reports/ for subdirectories named
YYYY-MM-DD_HH-MM-SS_<sha>, keeps the 5 most recent, and writes
<gh-pages-dir>/index.html.

Each report directory is expected to contain:
    branch                             — plain text file with the branch name
    cpp/coverage.html                  — gcovr HTML report
    cpp/coverage-summary.json          — gcovr JSON summary
    python/index.html                  — pytest-cov HTML report
    valgrind/vg_pytest/index.html      — ValgrindCI memcheck report (pytest)
    valgrind/vg_pytest/summary.txt     — valgrind-ci summary text
    valgrind/vg_pytest/raw.log         — raw valgrind text output
    valgrind/vg_pytest/raw.xml         — raw valgrind XML
    valgrind/vg_gtest/  (same layout)
    helgrind/hg_pytest/ (same layout)
    helgrind/hg_gtest/  (same layout)

Template files are resolved relative to this script's location:
    .github/ci/coverage/index.template.html
    .github/ci/coverage/style.css
"""

import json
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
    """Return line coverage % string from gcovr JSON summary, e.g. '64.6%'."""
    path = os.path.join(report_dir, "cpp", "coverage-summary.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pct = data.get("line_percent") or data.get("lines", {}).get("percent")
        if pct is not None:
            return f"{pct:.1f}%"
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def parse_python_coverage(report_dir: str) -> str | None:
    """Return total coverage % string from coverage.py HTML, e.g. '69%'."""
    html = _read(os.path.join(report_dir, "python", "index.html"))
    # coverage.py emits: <span class="pc_cov">69%</span>
    m = re.search(r'class=["\']pc_cov["\']>\s*([\d.]+%)\s*<', html)
    return m.group(1) if m else None


def parse_valgrind_errors(report_dir: str, subpath: str) -> str | None:
    """Return total error count from a ValgrindCI HTML report, e.g. '3 errors'.
    subpath is relative to report_dir, e.g. 'valgrind/vg_pytest'."""
    html = _read(os.path.join(report_dir, subpath, "index.html"))
    # ValgrindCI emits: <p><b>1</b> errors</p>
    m = re.search(r'<b>(\d+)</b>\s*errors?', html)
    if m:
        n = int(m.group(1))
        return f"{n} error{'s' if n != 1 else ''}"
    return None


def _has_memory_reports(report_dir: str) -> bool:
    """Return True if at least one memory report subdirectory exists."""
    for subpath in ("valgrind/vg_pytest", "valgrind/vg_gtest",
                    "helgrind/hg_pytest", "helgrind/hg_gtest"):
        if os.path.isdir(os.path.join(report_dir, subpath)):
            return True
    return False


def get_entries(reports_root: str) -> list[dict]:
    entries = []
    if not os.path.isdir(reports_root):
        return entries

    for name in sorted(os.listdir(reports_root), reverse=True):
        report_dir = os.path.join(reports_root, name)
        if not os.path.isdir(report_dir):
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_([0-9a-f]+)", name)
        if m:
            date_str = m.group(1)
            time_str = m.group(2).replace("-", ":")
            sha = m.group(3)
            utc_iso = f"{date_str}T{time_str}Z"
            label = f"{date_str} {time_str} UTC — {sha}"
        else:
            utc_iso = ""
            label = name
            sha = ""

        branch_file = os.path.join(report_dir, "branch")
        branch = _read(branch_file).strip() or None

        entries.append({
            "name": name,
            "label": label,
            "utc_iso": utc_iso,
            "sha": sha,
            "branch": branch,
            "has_memory": _has_memory_reports(report_dir),
            "cpp_coverage": parse_cpp_coverage(report_dir),
            "python_coverage": parse_python_coverage(report_dir),
            "valgrind_pytest_errors": parse_valgrind_errors(report_dir, "valgrind/vg_pytest"),
            "valgrind_gtest_errors":  parse_valgrind_errors(report_dir, "valgrind/vg_gtest"),
            "helgrind_pytest_errors": parse_valgrind_errors(report_dir, "helgrind/hg_pytest"),
            "helgrind_gtest_errors":  parse_valgrind_errors(report_dir, "helgrind/hg_gtest"),
            "cpp_path":    f"reports/{name}/cpp/coverage.html",
            "python_path": f"reports/{name}/python/index.html",
            # Memory sub-paths (base dir for each tool/suite combo)
            "vg_pytest_base":  f"reports/{name}/valgrind/vg_pytest",
            "vg_gtest_base":   f"reports/{name}/valgrind/vg_gtest",
            "hg_pytest_base":  f"reports/{name}/helgrind/hg_pytest",
            "hg_gtest_base":   f"reports/{name}/helgrind/hg_gtest",
        })
    return entries


def _stat_badge(value: str | None, is_error: bool = False) -> str:
    """Render a small inline stat next to a link button."""
    if value is None:
        return ""
    css_class = "stat-error" if is_error and value != "0 errors" else "stat-ok"
    return f'<span class="stat {css_class}">{value}</span>'


def _memory_group(title: str, base: str) -> str:
    """Render one memory-group block with links to summary.txt, raw.log, raw.xml."""
    return f"""
          <div class="memory-group">
            <div class="memory-group-title">{title}</div>
            <div class="memory-links">
              <a class="memory-link" href="{base}/summary.txt">
                <span class="memory-link-icon">&#x1f4cb;</span> summary.txt
              </a>
              <a class="memory-link" href="{base}/raw.log">
                <span class="memory-link-icon">&#x1f4c4;</span> raw.log
              </a>
              <a class="memory-link" href="{base}/raw.xml">
                <span class="memory-link-icon">&#x1f5c2;</span> raw.xml
              </a>
            </div>
          </div>"""


def render_cards(entries: list[dict]) -> str:
    if not entries:
        return '<div class="empty">No coverage reports found yet.</div>'

    cards = []
    for i, entry in enumerate(entries):
        is_latest = i == 0
        wrapper_class = "report-card-wrapper" + (" latest" if is_latest else "")
        badge = '<span class="badge">Latest</span>' if is_latest else ""
        branch_tag = (
            f'<div class="report-branch">&#x2387; {entry["branch"]}</div>'
            if entry.get("branch") else ""
        )

        cpp_stat    = _stat_badge(entry["cpp_coverage"])
        python_stat = _stat_badge(entry["python_coverage"])

        memory_toggle = ""
        memory_panel = ""
        if entry["has_memory"]:
            memory_toggle = """
      <button class="memory-toggle" aria-expanded="false">
        <span class="toggle-icon">&#x25b8;</span> Memory
      </button>"""

            groups = (
                _memory_group("Valgrind — pytest",  entry["vg_pytest_base"])
                + _memory_group("Valgrind — gtest",   entry["vg_gtest_base"])
                + _memory_group("Helgrind — pytest", entry["hg_pytest_base"])
                + _memory_group("Helgrind — gtest",  entry["hg_gtest_base"])
            )
            memory_panel = f"""
  <div class="memory-panel">
    <div class="memory-panel-inner">{groups}
    </div>
  </div>"""

        cards.append(f"""
<div class="{wrapper_class}">
  <div class="report-card">
    <span class="report-index">#{i + 1}</span>
    <div class="report-info">
      <div class="report-label" data-utc="{entry['utc_iso']}" data-sha="{entry['sha']}">{entry['label']}</div>
      {branch_tag}
    </div>
    {badge}
    <div class="report-links">
      <a class="report-link" href="{entry['cpp_path']}">C++ {cpp_stat}</a>
      <a class="report-link" href="{entry['python_path']}">Python {python_stat}</a>
    </div>{memory_toggle}
  </div>{memory_panel}
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
        print(f"Usage: {sys.argv[0]} <gh-pages-dir>", file=sys.stderr)
        sys.exit(1)

    gh_pages_dir = sys.argv[1]
    reports_root = os.path.join(gh_pages_dir, "reports")

    prune_old_reports(reports_root)
    entries = get_entries(reports_root)
    html = render_html(entries)

    out_path = os.path.join(gh_pages_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Written {out_path} with {len(entries)} report(s).")


if __name__ == "__main__":
    main()
