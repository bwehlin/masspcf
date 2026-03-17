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


def collect_coverage(
    dump_paths: str | list[str],
    mapping_path: str,
) -> list[LineCoverage]:
    """Combine counter dump(s) with mapping to produce per-line coverage data.

    Args:
        dump_paths: Path to a single binary counter dump, or a list of paths.
            When multiple paths are given, counters are summed element-wise.
        mapping_path: Path to mapping.json from the instrument step.
    """
    if isinstance(dump_paths, str):
        dump_paths = [dump_paths]

    counters = merge_counter_dumps(dump_paths)
    mapping = read_mapping(mapping_path)

    results = []
    for entry in mapping["mappings"]:
        cid = entry["id"]
        hit_count = counters[cid] if cid < len(counters) else 0
        results.append(LineCoverage(
            file=entry["file"],
            line=entry["line"],
            hit_count=hit_count,
        ))

    return results


def generate_lcov(coverage: list[LineCoverage], output_path: str) -> None:
    """Write coverage data in lcov .info format."""
    # Group by file
    by_file: dict[str, list[LineCoverage]] = defaultdict(list)
    for lc in coverage:
        by_file[lc.file].append(lc)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("TN:CUDA Coverage\n")

        for filepath in sorted(by_file):
            lines = sorted(by_file[filepath], key=lambda x: x.line)

            f.write(f"SF:{filepath}\n")

            for lc in lines:
                f.write(f"DA:{lc.line},{lc.hit_count}\n")

            total_lines = len(lines)
            hit_lines = sum(1 for lc in lines if lc.hit_count > 0)

            f.write(f"LF:{total_lines}\n")
            f.write(f"LH:{hit_lines}\n")
            f.write("end_of_record\n")

    print(f"lcov report written to {output_path}")


def generate_summary(coverage: list[LineCoverage], output_path: str) -> None:
    """Write a coverage-summary.json with per-file and aggregate line coverage."""
    total = len(coverage)
    hit = sum(1 for lc in coverage if lc.hit_count > 0)
    pct = (hit / total * 100) if total > 0 else 0.0

    # Per-file breakdown
    by_file: dict[str, list[LineCoverage]] = defaultdict(list)
    for lc in coverage:
        by_file[lc.file].append(lc)

    files = {}
    for filepath in sorted(by_file):
        lines = by_file[filepath]
        f_total = len(lines)
        f_hit = sum(1 for lc in lines if lc.hit_count > 0)
        files[filepath] = {
            "lines_total": f_total,
            "lines_covered": f_hit,
            "line_percent": round(f_hit / f_total * 100, 1) if f_total > 0 else 0.0,
        }

    summary = {
        "line_percent": round(pct, 1),
        "lines_total": total,
        "lines_covered": hit,
        "files": files,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Coverage summary: {hit}/{total} lines ({pct:.1f}%)")
    print(f"Summary written to {output_path}")


def collect_and_report(
    dump_path: str | list[str],
    mapping_path: str,
    lcov_path: str | None = None,
    summary_path: str | None = None,
) -> list[LineCoverage]:
    """Full pipeline: read dump(s) + mapping, produce all requested reports.

    Args:
        dump_path: Path to a single dump file, or a list of paths to merge.
    """
    coverage = collect_coverage(dump_path, mapping_path)

    if lcov_path:
        generate_lcov(coverage, lcov_path)

    if summary_path:
        generate_summary(coverage, summary_path)

    return coverage
