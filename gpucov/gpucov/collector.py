"""Collect CUDA coverage data and produce lcov/JSON reports.

Reads the binary counter dump produced by runtime.cuh's atexit handler
and the mapping.json produced by the instrumenter, then generates
lcov .info files and a coverage-summary.json.
"""

import json
import os
import struct
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class LineCoverage:
    """Coverage data for a single source line."""
    file: str
    line: int
    hit_count: int


@dataclass
class FunctionCoverage:
    """Coverage data for a single function."""
    name: str
    file: str
    line: int
    hit_count: int


@dataclass
class BranchCoverage:
    """Coverage data for a single branch point."""
    file: str
    line: int
    block_id: int
    true_count: int
    false_count: int


def read_counter_dump(dump_path: str) -> list[int]:
    """Read the binary counter dump file.

    Format: uint32 num_counters, then num_counters x uint32 values.
    """
    with open(dump_path, 'rb') as f:
        data = f.read()

    if len(data) < 4:
        raise ValueError(f"Counter dump too small: {len(data)} bytes")

    num_counters = struct.unpack('<I', data[:4])[0]
    expected_size = 4 + num_counters * 4

    if len(data) < expected_size:
        raise ValueError(
            f"Counter dump truncated: expected {expected_size} bytes, got {len(data)}"
        )

    counters = list(struct.unpack(f'<{num_counters}I', data[4:expected_size]))
    return counters


def read_mapping(mapping_path: str) -> dict:
    """Read the mapping.json file produced by the instrumenter."""
    with open(mapping_path, 'r') as f:
        return json.load(f)


def merge_counter_dumps(dump_paths: list[str]) -> list[int]:
    """Read multiple binary counter dumps and sum them element-wise.

    All dumps must have the same number of counters.
    """
    if not dump_paths:
        return []

    merged = read_counter_dump(dump_paths[0])

    for path in dump_paths[1:]:
        counters = read_counter_dump(path)
        if len(counters) != len(merged):
            raise ValueError(
                f"Counter count mismatch: {dump_paths[0]} has {len(merged)}, "
                f"{path} has {len(counters)}"
            )
        for i in range(len(merged)):
            merged[i] += counters[i]

    return merged


@dataclass
class CoverageResult:
    """Combined coverage data from all sources."""
    lines: list[LineCoverage]
    functions: list[FunctionCoverage]
    branches: list[BranchCoverage]


def collect_coverage(
    dump_paths: str | list[str],
    mapping_path: str,
) -> CoverageResult:
    """Combine counter dump(s) with mapping to produce coverage data.

    Args:
        dump_paths: Path to a single binary counter dump, or a list of paths.
            When multiple paths are given, counters are summed element-wise.
        mapping_path: Path to mapping.json from the instrument step.
    """
    if isinstance(dump_paths, str):
        dump_paths = [dump_paths]

    counters = merge_counter_dumps(dump_paths)
    mapping = read_mapping(mapping_path)

    def _count(cid: int) -> int:
        return counters[cid] if cid < len(counters) else 0

    lines = []
    for entry in mapping["mappings"]:
        lines.append(LineCoverage(
            file=entry["file"],
            line=entry["line"],
            hit_count=_count(entry["id"]),
        ))

    functions = []
    for entry in mapping.get("functions", []):
        functions.append(FunctionCoverage(
            name=entry["name"],
            file=entry["file"],
            line=entry["line"],
            hit_count=_count(entry["entry_counter_id"]),
        ))

    branches = []
    for entry in mapping.get("branches", []):
        branches.append(BranchCoverage(
            file=entry["file"],
            line=entry["line"],
            block_id=entry["block_id"],
            true_count=_count(entry["true_counter_id"]),
            false_count=_count(entry["false_counter_id"]),
        ))

    return CoverageResult(lines=lines, functions=functions, branches=branches)


def generate_lcov(result: CoverageResult, output_path: str) -> None:
    """Write coverage data in lcov .info format."""
    # Group everything by file
    lines_by_file: dict[str, list[LineCoverage]] = defaultdict(list)
    for lc in result.lines:
        lines_by_file[lc.file].append(lc)

    funcs_by_file: dict[str, list[FunctionCoverage]] = defaultdict(list)
    for fc in result.functions:
        funcs_by_file[fc.file].append(fc)

    branches_by_file: dict[str, list[BranchCoverage]] = defaultdict(list)
    for bc in result.branches:
        branches_by_file[bc.file].append(bc)

    all_files = sorted(set(lines_by_file) | set(funcs_by_file) | set(branches_by_file))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("TN:CUDA Coverage\n")

        for filepath in all_files:
            f.write(f"SF:{filepath}\n")

            # Function coverage
            funcs = sorted(funcs_by_file.get(filepath, []), key=lambda x: x.line)
            for fc in funcs:
                f.write(f"FN:{fc.line},{fc.name}\n")
            for fc in funcs:
                f.write(f"FNDA:{fc.hit_count},{fc.name}\n")
            if funcs:
                f.write(f"FNF:{len(funcs)}\n")
                f.write(f"FNH:{sum(1 for fc in funcs if fc.hit_count > 0)}\n")

            # Branch coverage
            branches = sorted(branches_by_file.get(filepath, []),
                              key=lambda x: (x.line, x.block_id))
            for bc in branches:
                tc = bc.true_count if bc.true_count > 0 else '-'
                fc = bc.false_count if bc.false_count > 0 else '-'
                f.write(f"BRDA:{bc.line},{bc.block_id},0,{tc}\n")
                f.write(f"BRDA:{bc.line},{bc.block_id},1,{fc}\n")
            if branches:
                brf = len(branches) * 2
                brh = sum(1 for bc in branches for c in (bc.true_count, bc.false_count) if c > 0)
                f.write(f"BRF:{brf}\n")
                f.write(f"BRH:{brh}\n")

            # Line coverage
            lines = sorted(lines_by_file.get(filepath, []), key=lambda x: x.line)
            for lc in lines:
                f.write(f"DA:{lc.line},{lc.hit_count}\n")
            f.write(f"LF:{len(lines)}\n")
            f.write(f"LH:{sum(1 for lc in lines if lc.hit_count > 0)}\n")

            f.write("end_of_record\n")

    print(f"lcov report written to {output_path}")


def _pct(hit: int, total: int) -> float:
    return round(hit / total * 100, 1) if total > 0 else 0.0


def generate_summary(result: CoverageResult, output_path: str) -> None:
    """Write a coverage-summary.json with per-file and aggregate coverage."""
    l_total = len(result.lines)
    l_hit = sum(1 for lc in result.lines if lc.hit_count > 0)

    fn_total = len(result.functions)
    fn_hit = sum(1 for fc in result.functions if fc.hit_count > 0)

    br_total = len(result.branches) * 2
    br_hit = sum(1 for bc in result.branches
                 for c in (bc.true_count, bc.false_count) if c > 0)

    # Per-file breakdown
    files: dict[str, dict] = {}
    all_file_keys = set()
    for lc in result.lines:
        all_file_keys.add(lc.file)
    for fc in result.functions:
        all_file_keys.add(fc.file)
    for bc in result.branches:
        all_file_keys.add(bc.file)

    for filepath in sorted(all_file_keys):
        fl = [lc for lc in result.lines if lc.file == filepath]
        ff = [fc for fc in result.functions if fc.file == filepath]
        fb = [bc for bc in result.branches if bc.file == filepath]
        fl_total, fl_hit = len(fl), sum(1 for lc in fl if lc.hit_count > 0)
        ff_total, ff_hit = len(ff), sum(1 for fc in ff if fc.hit_count > 0)
        fb_total = len(fb) * 2
        fb_hit = sum(1 for bc in fb for c in (bc.true_count, bc.false_count) if c > 0)
        files[filepath] = {
            "lines_total": fl_total, "lines_covered": fl_hit,
            "line_percent": _pct(fl_hit, fl_total),
            "functions_total": ff_total, "functions_covered": ff_hit,
            "function_percent": _pct(ff_hit, ff_total),
            "branches_total": fb_total, "branches_covered": fb_hit,
            "branch_percent": _pct(fb_hit, fb_total),
        }

    summary = {
        "line_percent": _pct(l_hit, l_total),
        "lines_total": l_total, "lines_covered": l_hit,
        "function_percent": _pct(fn_hit, fn_total),
        "functions_total": fn_total, "functions_covered": fn_hit,
        "branch_percent": _pct(br_hit, br_total),
        "branches_total": br_total, "branches_covered": br_hit,
        "files": files,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Coverage summary: {l_hit}/{l_total} lines ({_pct(l_hit, l_total)}%), "
          f"{fn_hit}/{fn_total} functions, {br_hit}/{br_total} branches")
    print(f"Summary written to {output_path}")


def collect_and_report(
    dump_path: str | list[str],
    mapping_path: str,
    lcov_path: str | None = None,
    summary_path: str | None = None,
) -> CoverageResult:
    """Full pipeline: read dump(s) + mapping, produce all requested reports.

    Args:
        dump_path: Path to a single dump file, or a list of paths to merge.
    """
    result = collect_coverage(dump_path, mapping_path)

    if lcov_path:
        generate_lcov(result, lcov_path)

    if summary_path:
        generate_summary(result, summary_path)

    return result
