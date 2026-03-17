"""Tests for gpucov.collector — counter dump reading and report generation."""

import json
import struct

import pytest

from gpucov.collector import (
    LineCoverage,
    collect_and_report,
    generate_lcov,
    generate_summary,
    merge_counter_dumps,
    read_counter_dump,
)


class TestReadCounterDump:
    def test_basic(self, tmp_path):
        counters = [10, 0, 5, 3]
        data = struct.pack('<I', len(counters))
        data += struct.pack(f'<{len(counters)}I', *counters)

        dump = tmp_path / "cov.bin"
        dump.write_bytes(data)

        assert read_counter_dump(str(dump)) == counters

    def test_single_counter(self, tmp_path):
        data = struct.pack('<II', 1, 42)
        dump = tmp_path / "cov.bin"
        dump.write_bytes(data)

        assert read_counter_dump(str(dump)) == [42]

    def test_zero_counters(self, tmp_path):
        data = struct.pack('<I', 0)
        dump = tmp_path / "cov.bin"
        dump.write_bytes(data)

        assert read_counter_dump(str(dump)) == []

    def test_truncated_raises(self, tmp_path):
        data = struct.pack('<I', 100)  # claims 100, has none
        dump = tmp_path / "cov.bin"
        dump.write_bytes(data)

        with pytest.raises(ValueError, match="truncated"):
            read_counter_dump(str(dump))

    def test_too_small_raises(self, tmp_path):
        dump = tmp_path / "cov.bin"
        dump.write_bytes(b"\x00")

        with pytest.raises(ValueError, match="too small"):
            read_counter_dump(str(dump))


def _write_dump(path, counters):
    """Helper: write a binary counter dump file."""
    data = struct.pack('<I', len(counters))
    data += struct.pack(f'<{len(counters)}I', *counters)
    path.write_bytes(data)


class TestMergeCounterDumps:
    def test_single_file(self, tmp_path):
        d = tmp_path / "a.bin"
        _write_dump(d, [1, 2, 3])
        assert merge_counter_dumps([str(d)]) == [1, 2, 3]

    def test_two_files_summed(self, tmp_path):
        _write_dump(tmp_path / "a.bin", [10, 0, 5])
        _write_dump(tmp_path / "b.bin", [3, 7, 0])
        result = merge_counter_dumps([str(tmp_path / "a.bin"), str(tmp_path / "b.bin")])
        assert result == [13, 7, 5]

    def test_three_files(self, tmp_path):
        _write_dump(tmp_path / "a.bin", [1, 0])
        _write_dump(tmp_path / "b.bin", [0, 1])
        _write_dump(tmp_path / "c.bin", [2, 3])
        result = merge_counter_dumps([
            str(tmp_path / "a.bin"),
            str(tmp_path / "b.bin"),
            str(tmp_path / "c.bin"),
        ])
        assert result == [3, 4]

    def test_mismatched_counter_count_raises(self, tmp_path):
        _write_dump(tmp_path / "a.bin", [1, 2, 3])
        _write_dump(tmp_path / "b.bin", [1, 2])
        with pytest.raises(ValueError, match="mismatch"):
            merge_counter_dumps([str(tmp_path / "a.bin"), str(tmp_path / "b.bin")])

    def test_empty_list(self):
        assert merge_counter_dumps([]) == []


class TestGenerateLcov:
    def test_multiple_files(self, tmp_path):
        coverage = [
            LineCoverage(file="/src/a.cuh", line=10, hit_count=5),
            LineCoverage(file="/src/a.cuh", line=11, hit_count=0),
            LineCoverage(file="/src/b.cuh", line=20, hit_count=3),
        ]

        lcov_path = str(tmp_path / "cuda.info")
        generate_lcov(coverage, lcov_path)

        content = (tmp_path / "cuda.info").read_text()
        assert "TN:CUDA Coverage" in content
        assert "SF:/src/a.cuh" in content
        assert "DA:10,5" in content
        assert "DA:11,0" in content
        assert "LF:2" in content  # 2 lines in a.cuh
        assert "LH:1" in content  # 1 hit in a.cuh
        assert "SF:/src/b.cuh" in content
        assert "DA:20,3" in content
        assert content.count("end_of_record") == 2

    def test_empty_coverage(self, tmp_path):
        lcov_path = str(tmp_path / "empty.info")
        generate_lcov([], lcov_path)

        content = (tmp_path / "empty.info").read_text()
        assert "TN:CUDA Coverage" in content
        assert "end_of_record" not in content

    def test_lines_sorted_by_number(self, tmp_path):
        coverage = [
            LineCoverage(file="/f.cuh", line=30, hit_count=1),
            LineCoverage(file="/f.cuh", line=10, hit_count=2),
            LineCoverage(file="/f.cuh", line=20, hit_count=3),
        ]

        lcov_path = str(tmp_path / "sorted.info")
        generate_lcov(coverage, lcov_path)

        content = (tmp_path / "sorted.info").read_text()
        lines = [l for l in content.splitlines() if l.startswith("DA:")]
        assert lines == ["DA:10,2", "DA:20,3", "DA:30,1"]


class TestGenerateSummary:
    def test_basic(self, tmp_path):
        coverage = [
            LineCoverage(file="/src/a.cuh", line=10, hit_count=5),
            LineCoverage(file="/src/a.cuh", line=11, hit_count=0),
            LineCoverage(file="/src/a.cuh", line=12, hit_count=1),
            LineCoverage(file="/src/b.cuh", line=20, hit_count=0),
        ]

        summary_path = str(tmp_path / "summary.json")
        generate_summary(coverage, summary_path)

        data = json.loads((tmp_path / "summary.json").read_text())
        assert data["lines_total"] == 4
        assert data["lines_covered"] == 2
        assert data["line_percent"] == 50.0
        assert data["files"]["/src/a.cuh"]["lines_total"] == 3
        assert data["files"]["/src/a.cuh"]["lines_covered"] == 2
        assert data["files"]["/src/b.cuh"]["line_percent"] == 0.0

    def test_empty(self, tmp_path):
        summary_path = str(tmp_path / "summary.json")
        generate_summary([], summary_path)

        data = json.loads((tmp_path / "summary.json").read_text())
        assert data["lines_total"] == 0
        assert data["lines_covered"] == 0
        assert data["line_percent"] == 0.0

    def test_all_covered(self, tmp_path):
        coverage = [
            LineCoverage(file="/f.cuh", line=1, hit_count=1),
            LineCoverage(file="/f.cuh", line=2, hit_count=99),
        ]

        summary_path = str(tmp_path / "summary.json")
        generate_summary(coverage, summary_path)

        data = json.loads((tmp_path / "summary.json").read_text())
        assert data["line_percent"] == 100.0


class TestCollectAndReport:
    def _make_dump_and_mapping(self, tmp_path, counters, mappings):
        dump_data = struct.pack('<I', len(counters))
        dump_data += struct.pack(f'<{len(counters)}I', *counters)
        (tmp_path / "cov.bin").write_bytes(dump_data)

        mapping = {"num_counters": len(counters), "mappings": mappings}
        (tmp_path / "mapping.json").write_text(json.dumps(mapping))

    def test_roundtrip(self, tmp_path):
        self._make_dump_and_mapping(tmp_path, [42, 0, 7], [
            {"id": 0, "file": "/src/kernel.cuh", "line": 10},
            {"id": 1, "file": "/src/kernel.cuh", "line": 15},
            {"id": 2, "file": "/src/ops.cuh", "line": 5},
        ])

        coverage = collect_and_report(
            dump_path=str(tmp_path / "cov.bin"),
            mapping_path=str(tmp_path / "mapping.json"),
            lcov_path=str(tmp_path / "cuda.info"),
            summary_path=str(tmp_path / "summary.json"),
        )

        assert len(coverage) == 3
        assert coverage[0].hit_count == 42
        assert coverage[1].hit_count == 0
        assert coverage[2].hit_count == 7

        assert (tmp_path / "cuda.info").exists()
        assert "DA:10,42" in (tmp_path / "cuda.info").read_text()

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["lines_total"] == 3
        assert summary["lines_covered"] == 2

    def test_counter_id_out_of_range_gives_zero(self, tmp_path):
        """Counter ID beyond the dump array should report 0 hits."""
        self._make_dump_and_mapping(tmp_path, [5], [
            {"id": 0, "file": "/f.cuh", "line": 1},
            {"id": 99, "file": "/f.cuh", "line": 2},  # out of range
        ])

        coverage = collect_and_report(
            dump_path=str(tmp_path / "cov.bin"),
            mapping_path=str(tmp_path / "mapping.json"),
            summary_path=str(tmp_path / "summary.json"),
        )

        assert coverage[0].hit_count == 5
        assert coverage[1].hit_count == 0

    def test_multiple_dumps_merged(self, tmp_path):
        """collect_and_report with a list of dump files sums counters."""
        mappings = [
            {"id": 0, "file": "/f.cuh", "line": 1},
            {"id": 1, "file": "/f.cuh", "line": 2},
        ]
        mapping = {"num_counters": 2, "mappings": mappings}
        (tmp_path / "mapping.json").write_text(json.dumps(mapping))

        # First run: line 1 hit, line 2 not
        _write_dump(tmp_path / "run1.bin", [10, 0])
        # Second run: line 1 not hit, line 2 hit
        _write_dump(tmp_path / "run2.bin", [0, 5])

        coverage = collect_and_report(
            dump_path=[str(tmp_path / "run1.bin"), str(tmp_path / "run2.bin")],
            mapping_path=str(tmp_path / "mapping.json"),
            summary_path=str(tmp_path / "summary.json"),
        )

        assert coverage[0].hit_count == 10
        assert coverage[1].hit_count == 5

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["lines_covered"] == 2
        assert summary["line_percent"] == 100.0
