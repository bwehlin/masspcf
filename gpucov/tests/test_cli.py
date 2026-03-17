"""Tests for gpucov CLI (argument parsing and subcommands)."""

import json
import struct
import sys

import pytest

from gpucov.__main__ import main


class TestGlobalOptions:
    def test_cmake_dir_prints_path(self, capsys):
        sys.argv = ['gpucov', '--cmake-dir']
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "cmake" in captured.out
        assert "GPUCovConfig.cmake" not in captured.out  # prints dir, not file

    def test_no_subcommand_prints_help(self, capsys):
        sys.argv = ['gpucov']
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestInstrumentSubcommand:
    def test_requires_source_root(self):
        sys.argv = ['gpucov', 'instrument', '--output-dir', '/tmp', '--files', 'a.cu']
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_requires_output_dir(self):
        sys.argv = ['gpucov', 'instrument', '--source-root', '.', '--files', 'a.cu']
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_requires_files(self):
        sys.argv = ['gpucov', 'instrument', '--source-root', '.', '--output-dir', '/tmp']
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0


class TestCollectSubcommand:
    def test_requires_dump(self):
        sys.argv = ['gpucov', 'collect', '--mapping', '/tmp/m.json', '--lcov', '/tmp/o.info']
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_requires_mapping(self):
        sys.argv = ['gpucov', 'collect', '--dump', '/tmp/d.bin', '--lcov', '/tmp/o.info']
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_requires_at_least_one_output(self, tmp_path):
        """collect with --dump and --mapping but no --lcov or --summary should fail."""
        dump = tmp_path / "dump.bin"
        dump.write_bytes(struct.pack('<I', 0))
        mapping = tmp_path / "mapping.json"
        mapping.write_text(json.dumps({"num_counters": 0, "mappings": []}))

        sys.argv = [
            'gpucov', 'collect',
            '--dump', str(dump),
            '--mapping', str(mapping),
        ]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_successful_collect(self, tmp_path):
        """Full collect with valid data should exit 0."""
        counters = [10, 0]
        dump_data = struct.pack('<I', 2) + struct.pack('<2I', *counters)
        (tmp_path / "dump.bin").write_bytes(dump_data)
        (tmp_path / "mapping.json").write_text(json.dumps({
            "num_counters": 2,
            "mappings": [
                {"id": 0, "file": "/f.cuh", "line": 1},
                {"id": 1, "file": "/f.cuh", "line": 2},
            ],
        }))

        sys.argv = [
            'gpucov', 'collect',
            '--dump', str(tmp_path / "dump.bin"),
            '--mapping', str(tmp_path / "mapping.json"),
            '--summary', str(tmp_path / "summary.json"),
        ]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["lines_covered"] == 1


class TestZerocountersSubcommand:
    def test_removes_matching_files(self, tmp_path):
        (tmp_path / "cuda_100.bin").write_bytes(b"\x00" * 8)
        (tmp_path / "cuda_200.bin").write_bytes(b"\x00" * 8)
        (tmp_path / "unrelated.txt").write_text("keep me")

        sys.argv = [
            'gpucov', 'zerocounters',
            '--dump', str(tmp_path / "cuda_*.bin"),
        ]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

        assert not (tmp_path / "cuda_100.bin").exists()
        assert not (tmp_path / "cuda_200.bin").exists()
        assert (tmp_path / "unrelated.txt").exists()

    def test_no_matches_is_not_an_error(self, tmp_path):
        sys.argv = [
            'gpucov', 'zerocounters',
            '--dump', str(tmp_path / "nonexistent_*.bin"),
        ]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_multiple_patterns(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"\x00")
        (tmp_path / "b.bin").write_bytes(b"\x00")

        sys.argv = [
            'gpucov', 'zerocounters',
            '--dump', str(tmp_path / "a.bin"), str(tmp_path / "b.bin"),
        ]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

        assert not (tmp_path / "a.bin").exists()
        assert not (tmp_path / "b.bin").exists()
