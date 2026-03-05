#!/usr/bin/env python3
"""
Generates index.html for the gh-pages coverage history site, plus a
per-run report.html detail page (with sidebar nav + iframe) for each run.

Usage:
    python generate_coverage_index.py <gh-pages-dir>

The script scans <gh-pages-dir>/reports/ for subdirectories named
YYYY-MM-DD_HH-MM-SS_<sha>, keeps the 5 most recent, and writes:
  <gh-pages-dir>/index.html
  <gh-pages-dir>/reports/<run>/report.html   (one per run)

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
    .github/ci/coverage/report.template.html
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


def load_template(name: str) -> str:
    path = os.path.join(TEMPLATES_DIR, name)
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
    m = re.search(r'class=["\']pc_cov["\']>\s*([\d.]+%)\s*<', html)
    return m.group(1) if m else None


def parse_valgrind_error_count(report_dir: str, subpath: str) -> int | None:
    """Return integer error count from a ValgrindCI HTML report, or None if unavailable."""
    html = _read(os.path.join(report_dir, subpath, "index.html"))
    m = re.search(r'<b>(\d+)</b>\s*errors?', html)
    return int(m.group(1)) if m else None


def _has_memory_reports(report_dir: str) -> bool:
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
            label = f"{date_str} {time_str} UTC \u2014 {sha}"
        else:
            utc_iso = ""
            label = name
            sha = ""

        branch = _read(os.path.join(report_dir, "branch")).strip() or None

        vg_pytest_errs  = parse_valgrind_error_count(report_dir, "valgrind/vg_pytest")
        vg_gtest_errs   = parse_valgrind_error_count(report_dir, "valgrind/vg_gtest")
        hg_pytest_errs  = parse_valgrind_error_count(report_dir, "helgrind/hg_pytest")
        hg_gtest_errs   = parse_valgrind_error_count(report_dir, "helgrind/hg_gtest")

        entries.append({
            "name": name,
            "label": label,
            "utc_iso": utc_iso,
            "sha": sha,
            "branch": branch,
            "has_memory": _has_memory_reports(report_dir),
            "cpp_coverage": parse_cpp_coverage(report_dir),
            "python_coverage": parse_python_coverage(report_dir),
            "vg_pytest_errors": vg_pytest_errs,
            "vg_gtest_errors":  vg_gtest_errs,
            "hg_pytest_errors": hg_pytest_errs,
            "hg_gtest_errors":  hg_gtest_errs,
            # Paths relative to gh-pages root (used in index.html)
            "detail_path":  f"reports/{name}/report.html",
            "cpp_path":     f"reports/{name}/cpp/coverage.html",
            "python_path":  f"reports/{name}/python/index.html",
            # Paths relative to the run dir (used in report.html iframe srcs)
            "cpp_rel":             "cpp/coverage.html",
            "python_rel":          "python/index.html",
            "vg_pytest_rel_base":  "valgrind/vg_pytest",
            "vg_gtest_rel_base":   "valgrind/vg_gtest",
            "hg_pytest_rel_base":  "helgrind/hg_pytest",
            "hg_gtest_rel_base":   "helgrind/hg_gtest",
        })
    return entries


# ── Badge helpers ──────────────────────────────────────────────────────────────

def _cov_badge_index(value: str | None) -> str:
    """Small coverage % badge for the index page cards."""
    if value is None:
        return ""
    return f'<span class="stat stat-ok">{value}</span>'


def _err_badge_index(count: int | None) -> str:
    """Small error count badge for the index page cards."""
    if count is None:
        return ""
    label = f"{count} err{'s' if count != 1 else ''}"
    cls = "stat-error" if count > 0 else "stat-ok"
    return f'<span class="stat {cls}">{label}</span>'


def _cov_badge_sidebar(value: str | None) -> str:
    """Coverage badge for the detail page sidebar."""
    if value is None:
        return ""
    return f'<span class="sidebar-section-badge badge-ok">{value}</span>'


def _err_badge_sidebar(count: int | None) -> str:
    """Error count badge for the detail page sidebar nav items."""
    if count is None:
        return ""
    cls = "badge-err" if count > 0 else "badge-ok"
    label = f"{count} err{'s' if count != 1 else ''}"
    return f'<span class="sidebar-section-badge {cls}">{label}</span>'


def _mem_section_badge(counts: list[int | None]) -> str:
    """Aggregate badge for a whole memory section header."""
    known = [c for c in counts if c is not None]
    if not known:
        return '<span class="sidebar-section-badge badge-none">—</span>'
    total = sum(known)
    cls = "badge-err" if total > 0 else "badge-ok"
    label = f"{total} err{'s' if total != 1 else ''}"
    return f'<span class="sidebar-section-badge {cls}">{label}</span>'


# ── Index page rendering ───────────────────────────────────────────────────────

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

        # Total memory errors for a compact summary badge on the card
        mem_counts = [entry["vg_pytest_errors"], entry["vg_gtest_errors"],
                      entry["hg_pytest_errors"], entry["hg_gtest_errors"]]
        known = [c for c in mem_counts if c is not None]
        if known:
            total_mem = sum(known)
            mem_cls = "stat-error" if total_mem > 0 else "stat-ok"
            mem_label = f"{total_mem} mem err{'s' if total_mem != 1 else ''}"
            mem_stat = f'<span class="stat {mem_cls}">{mem_label}</span>'
        else:
            mem_stat = ""

        cards.append(f"""
<div class="{wrapper_class}">
  <a class="report-card-link" href="{entry['detail_path']}">
    <div class="report-card">
      <span class="report-index">#{i + 1}</span>
      <div class="report-info">
        <div class="report-label" data-utc="{entry['utc_iso']}" data-sha="{entry['sha']}">{entry['label']}</div>
        {branch_tag}
      </div>
      {badge}
      <div class="report-stats">
        {_cov_badge_index(entry['cpp_coverage']) and f'<span class="stat-group"><span class="stat-label">C++</span>{_cov_badge_index(entry["cpp_coverage"])}</span>' or ''}
        {_cov_badge_index(entry['python_coverage']) and f'<span class="stat-group"><span class="stat-label">Py</span>{_cov_badge_index(entry["python_coverage"])}</span>' or ''}
        {mem_stat}
      </div>
      <div class="card-arrow">&#x2192;</div>
    </div>
  </a>
</div>""")

    return "\n".join(cards)


def render_index_html(entries: list[dict]) -> str:
    template = load_template("index.template.html")
    css = load_template("style.css")
    cards = render_cards(entries)
    return (template
            .replace("{{style}}", css)
            .replace("{{cards}}", cards)
            .replace("{{max_reports}}", str(MAX_REPORTS)))


# ── Detail page rendering ──────────────────────────────────────────────────────

def _memory_nav_section(entry: dict) -> str:
    """Render the Valgrind + Helgrind sidebar sections for report.html."""
    if not entry["has_memory"]:
        return ""

    def _group_items(base: str, errors: int | None) -> str:
        err_badge = _err_badge_sidebar(errors)
        return f"""
        <a class="sidebar-nav-item" data-src="{base}/index.html" href="#">
          <span class="nav-icon">&#x1f4ca;</span> HTML report {err_badge}
        </a>
        <a class="sidebar-nav-item" data-src="{base}/summary.txt" href="#">
          <span class="nav-icon">&#x1f4cb;</span> summary.txt
        </a>
        <a class="sidebar-nav-item" data-src="{base}/raw.log" href="#">
          <span class="nav-icon">&#x1f4c4;</span> raw.log
        </a>
        <a class="sidebar-nav-item" data-src="{base}/raw.xml" href="#">
          <span class="nav-icon">&#x1f5c2;</span> raw.xml
        </a>"""

    vg_counts = [entry["vg_pytest_errors"], entry["vg_gtest_errors"]]
    hg_counts = [entry["hg_pytest_errors"], entry["hg_gtest_errors"]]

    return f"""
    <!-- Valgrind section -->
    <div class="sidebar-section open">
      <div class="sidebar-section-header">
        <span class="sidebar-section-title">Valgrind (memcheck)</span>
        {_mem_section_badge(vg_counts)}
        <span class="sidebar-chevron">&#x25b8;</span>
      </div>
      <div class="sidebar-section-body">
        <div class="sidebar-subsection-label">pytest</div>
        {_group_items(entry['vg_pytest_rel_base'], entry['vg_pytest_errors'])}
        <div class="sidebar-subsection-label">gtest</div>
        {_group_items(entry['vg_gtest_rel_base'], entry['vg_gtest_errors'])}
      </div>
    </div>

    <!-- Helgrind section -->
    <div class="sidebar-section open">
      <div class="sidebar-section-header">
        <span class="sidebar-section-title">Helgrind</span>
        {_mem_section_badge(hg_counts)}
        <span class="sidebar-chevron">&#x25b8;</span>
      </div>
      <div class="sidebar-section-body">
        <div class="sidebar-subsection-label">pytest</div>
        {_group_items(entry['hg_pytest_rel_base'], entry['hg_pytest_errors'])}
        <div class="sidebar-subsection-label">gtest</div>
        {_group_items(entry['hg_gtest_rel_base'], entry['hg_gtest_errors'])}
      </div>
    </div>"""


def render_detail_html(entry: dict) -> str:
    template = load_template("report.template.html")
    branch_display = f"&#x2387; {entry['branch']}" if entry.get("branch") else ""

    return (template
            .replace("{{page_title}}", f"Report \u2014 {entry['sha'] or entry['name']}")
            .replace("{{utc_iso}}", entry["utc_iso"])
            .replace("{{sha}}", entry["sha"])
            .replace("{{label}}", entry["label"])
            .replace("{{branch_display}}", branch_display)
            .replace("{{cpp_path}}", entry["cpp_rel"])
            .replace("{{python_path}}", entry["python_rel"])
            .replace("{{cpp_badge}}", _cov_badge_sidebar(entry["cpp_coverage"]))
            .replace("{{python_badge}}", _cov_badge_sidebar(entry["python_coverage"]))
            .replace("{{memory_nav}}", _memory_nav_section(entry)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gh-pages-dir>", file=sys.stderr)
        sys.exit(1)

    gh_pages_dir = sys.argv[1]
    reports_root = os.path.join(gh_pages_dir, "reports")

    prune_old_reports(reports_root)
    entries = get_entries(reports_root)

    # Write per-run detail pages
    for entry in entries:
        detail_html = render_detail_html(entry)
        out_path = os.path.join(gh_pages_dir, "reports", entry["name"], "report.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(detail_html)
        print(f"  Written detail page: {out_path}")

    # Write main index
    index_html = render_index_html(entries)
    out_path = os.path.join(gh_pages_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    print(f"Written {out_path} with {len(entries)} report(s).")


if __name__ == "__main__":
    main()
