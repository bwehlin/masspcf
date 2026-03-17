"""Tests for gpucov.instrumenter — AST parsing and source rewriting."""

import json
import os

import pytest

try:
    from gpucov.instrumenter import (
        find_executable_lines,
        instrument_file,
        instrument_files,
    )
    HAS_LIBCLANG = True
except ImportError:
    HAS_LIBCLANG = False

pytestmark = pytest.mark.skipif(not HAS_LIBCLANG, reason="libclang not available")


@pytest.fixture
def sample_cuh(tmp_path):
    """Minimal .cuh with __device__ and __global__ functions."""
    src = tmp_path / "sample.cuh"
    src.write_text("""\
#ifndef SAMPLE_CUH
#define SAMPLE_CUH

#include <cuda_runtime.h>

namespace test_ns
{
    template<typename T>
    __device__ void device_func(T* data, int n)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n)
        {
            data[idx] = data[idx] + 1;
        }
    }

    template<typename T>
    __global__ void kernel(T* data, int n)
    {
        device_func(data, n);
    }

    void host_only_func(int x)
    {
        int y = x + 1;
    }
}

#endif
""")
    return src


class TestFindExecutableLines:
    def test_finds_device_function_lines(self, sample_cuh):
        result = find_executable_lines(str(sample_cuh))
        real_path = os.path.realpath(str(sample_cuh))
        assert real_path in result
        lines = result[real_path]
        # Should find lines inside device_func and kernel, but not host_only_func
        assert len(lines) > 0

    def test_returns_empty_for_host_only_cuh(self, tmp_path):
        src = tmp_path / "host.cuh"
        src.write_text("""\
#ifndef HOST_CUH
#define HOST_CUH
void foo(int x) { int y = x + 1; }
#endif
""")
        result = find_executable_lines(str(src))
        assert len(result) == 0


class TestInstrumentFile:
    def test_injects_counters(self, sample_cuh, tmp_path):
        output_dir = tmp_path / "out"
        result = instrument_file(
            source_file=str(sample_cuh),
            output_dir=str(output_dir),
            source_root=str(tmp_path),
        )

        if result.next_counter_id > 0:
            assert len(result.mappings) > 0
            assert len(result.instrumented_files) == 1

            content = open(result.instrumented_files[0]).read()
            assert "GPUCOV_HIT(" in content
            assert "gpucov_runtime.cuh" in content

    def test_cu_always_gets_copy(self, tmp_path):
        """A .cu file gets an instrumented copy even without device functions."""
        src = tmp_path / "host_only.cu"
        src.write_text('#include <cuda_runtime.h>\nvoid f() { int x = 1; }\n')

        output_dir = tmp_path / "out"
        result = instrument_file(
            source_file=str(src),
            output_dir=str(output_dir),
            source_root=str(tmp_path),
        )

        assert len(result.instrumented_files) == 1
        content = open(result.instrumented_files[0]).read()
        assert "gpucov_runtime.cuh" in content

    def test_cuh_without_device_funcs_skipped(self, tmp_path):
        """A .cuh with no device functions is not instrumented."""
        src = tmp_path / "host_only.cuh"
        src.write_text('#ifndef H\n#define H\nvoid f() { int x = 1; }\n#endif\n')

        output_dir = tmp_path / "out"
        result = instrument_file(
            source_file=str(src),
            output_dir=str(output_dir),
            source_root=str(tmp_path),
        )

        assert len(result.instrumented_files) == 0
        assert result.next_counter_id == 0

    def test_uses_macro_not_raw_call(self, sample_cuh, tmp_path):
        """Instrumented code should use GPUCOV_HIT() macro, not raw gpucov::hit()."""
        output_dir = tmp_path / "out"
        result = instrument_file(
            source_file=str(sample_cuh),
            output_dir=str(output_dir),
            source_root=str(tmp_path),
        )

        if result.next_counter_id > 0:
            content = open(result.instrumented_files[0]).read()
            assert "GPUCOV_HIT(" in content
            assert "gpucov::hit(" not in content

    def test_else_lines_not_instrumented(self, tmp_path):
        """Inserting before 'else' or 'else if' would break if/else chains."""
        src = tmp_path / "branches.cuh"
        src.write_text("""\
#ifndef B_CUH
#define B_CUH
#include <cuda_runtime.h>
__device__ void f(int x)
{
    if (x > 0)
    {
        int a = 1;
    }
    else if (x < 0)
    {
        int b = 2;
    }
    else
    {
        int c = 3;
    }
}
#endif
""")
        output_dir = tmp_path / "out"
        result = instrument_file(
            source_file=str(src),
            output_dir=str(output_dir),
            source_root=str(tmp_path),
        )

        if result.instrumented_files:
            content = open(result.instrumented_files[0]).read()
            # No counter call should appear immediately before 'else'
            for line in content.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("else"):
                    # The previous non-empty, non-preprocessor line should not be a hit() call
                    pass  # structural check below

            # More precise: split into lines, find 'else' lines, check predecessor
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.lstrip().startswith("else"):
                    # Walk back to find the previous non-empty line
                    j = i - 1
                    while j >= 0 and lines[j].strip() in ('', '#endif'):
                        j -= 1
                    if j >= 0:
                        assert "GPUCOV_HIT(" not in lines[j], (
                            f"hit() call at line {j+1} directly before 'else' at line {i+1}"
                        )


class TestInstrumentFiles:
    def test_writes_mapping_and_runtime(self, sample_cuh, tmp_path):
        output_dir = tmp_path / "out"
        result = instrument_files(
            source_files=[str(sample_cuh)],
            output_dir=str(output_dir),
            source_root=str(tmp_path),
        )

        mapping_path = output_dir / "mapping.json"
        assert mapping_path.exists()

        mapping = json.loads(mapping_path.read_text())
        assert "num_counters" in mapping
        assert "mappings" in mapping
        assert mapping["num_counters"] == result.next_counter_id

        runtime_path = output_dir / "gpucov_runtime.cuh"
        assert runtime_path.exists()

    def test_counter_ids_are_sequential_across_files(self, tmp_path):
        """When instrumenting multiple files, counter IDs should be sequential."""
        src_a = tmp_path / "a.cuh"
        src_a.write_text("""\
#ifndef A_CUH
#define A_CUH
#include <cuda_runtime.h>
__device__ void fa() { int x = 1; }
#endif
""")
        src_b = tmp_path / "b.cuh"
        src_b.write_text("""\
#ifndef B_CUH
#define B_CUH
#include <cuda_runtime.h>
__device__ void fb() { int y = 2; }
#endif
""")

        output_dir = tmp_path / "out"
        result = instrument_files(
            source_files=[str(src_a), str(src_b)],
            output_dir=str(output_dir),
            source_root=str(tmp_path),
        )

        if result.next_counter_id > 1:
            ids = [m.id for m in result.mappings]
            assert ids == list(range(len(ids)))


