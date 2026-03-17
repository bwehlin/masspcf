"""libclang-based CUDA source instrumenter.

Parses .cu/.cuh files, identifies executable lines inside __global__,
__device__, and __host__ __device__ functions, and injects atomicAdd
counter increments for coverage tracking.
"""

import glob as globmod
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from clang.cindex import (
    Config,
    CursorKind,
    Index,
    TranslationUnit,
)

# Cursor kinds that represent executable statements we want to instrument
_EXECUTABLE_KINDS = {
    CursorKind.IF_STMT,
    CursorKind.FOR_STMT,
    CursorKind.WHILE_STMT,
    CursorKind.DO_STMT,
    CursorKind.RETURN_STMT,
    CursorKind.CALL_EXPR,
    CursorKind.BINARY_OPERATOR,
    CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
    CursorKind.UNARY_OPERATOR,
    CursorKind.CXX_UNARY_EXPR,
    CursorKind.DECL_STMT,
    CursorKind.SWITCH_STMT,
    CursorKind.CASE_STMT,
    CursorKind.MEMBER_REF_EXPR,
}

# CUDA attributes that mark device-side functions
_CUDA_ATTR_PATTERN = re.compile(r'__(?:global|device)__')

# Runtime include file name
RUNTIME_HEADER = "gpucov_runtime.cuh"

# Namespace for injected counter calls
COUNTER_NAMESPACE = "gpucov"

# Compile definition used to guard instrumented code
COVERAGE_DEFINE = "GPUCOV_ENABLED"


@dataclass
class CounterMapping:
    """Maps a counter ID to a source file and line number."""
    id: int
    file: str
    line: int

    def to_dict(self) -> dict:
        return {"id": self.id, "file": self.file, "line": self.line}


@dataclass
class InstrumentationResult:
    """Result of instrumenting a set of source files."""
    mappings: list[CounterMapping] = field(default_factory=list)
    instrumented_files: list[str] = field(default_factory=list)
    next_counter_id: int = 0


def _find_cuda_functions(cursor, source_file: str) -> list:
    """Walk the AST and find functions with CUDA attributes in the given file."""
    results = []

    def _walk(node):
        # Only process nodes from the file we're instrumenting
        if node.location.file and os.path.realpath(node.location.file.name) != source_file:
            return

        if node.kind in (
            CursorKind.FUNCTION_DECL,
            CursorKind.FUNCTION_TEMPLATE,
            CursorKind.CXX_METHOD,
        ):
            if _is_cuda_function(node, source_file):
                results.append(node)

        for child in node.get_children():
            _walk(child)

    _walk(cursor)
    return results


def _is_cuda_function(node, source_file: str) -> bool:
    """Check if a function node has CUDA __global__, __device__, or __host__ __device__ attributes."""
    if not node.location.file:
        return False

    # Check extent — read the raw source text for CUDA qualifiers
    start = node.extent.start
    end = node.extent.end

    if not start.file or os.path.realpath(start.file.name) != source_file:
        return False

    try:
        with open(source_file, 'r') as f:
            content = f.read()

        # Get text from the start of the function declaration to the opening brace
        line_offsets = [0]
        for i, ch in enumerate(content):
            if ch == '\n':
                line_offsets.append(i + 1)

        start_offset = line_offsets[start.line - 1] + start.column - 1
        # Look back a bit for attributes that precede the return type
        search_start = max(0, start_offset - 200)
        decl_text = content[search_start:start_offset + 200]

        return bool(_CUDA_ATTR_PATTERN.search(decl_text))
    except (OSError, IndexError):
        return False


def _collect_executable_lines(node, source_file: str) -> set[int]:
    """Collect line numbers of executable statements within a function body."""
    lines = set()

    def _walk(n):
        if n.location.file and os.path.realpath(n.location.file.name) == source_file:
            if n.kind in _EXECUTABLE_KINDS:
                lines.add(n.location.line)

        for child in n.get_children():
            _walk(child)

    # Find the compound statement (function body)
    for child in node.get_children():
        if child.kind == CursorKind.COMPOUND_STMT:
            _walk(child)
            break

    return lines


def _get_system_include_args() -> list[str]:
    """Detect GCC/system C++ include paths for libclang.

    libclang's bundled library doesn't include standard C++ headers,
    so we need to find them from the system compiler.
    """
    args = []

    # Try to get GCC's include paths
    try:
        result = subprocess.run(
            ['g++', '-E', '-x', 'c++', '-', '-v'],
            input='', capture_output=True, text=True,
        )
        stderr = result.stderr
        # Parse the #include <...> search list
        in_search = False
        for line in stderr.splitlines():
            if '#include <...> search starts here:' in line:
                in_search = True
                continue
            if 'End of search list.' in line:
                break
            if in_search:
                path = line.strip()
                if os.path.isdir(path):
                    args.extend(['-isystem', path])
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Fallback: find GCC includes by glob
    if not args:
        for pattern in ['/usr/include/c++/*', '/usr/include/x86_64-linux-gnu/c++/*']:
            for p in sorted(globmod.glob(pattern), reverse=True):
                if os.path.isdir(p):
                    args.extend(['-isystem', p])
                    break

    return args


def find_executable_lines(
    source_file: str,
    include_paths: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, set[int]]:
    """Parse a CUDA source file and return executable lines in device functions.

    Returns a dict mapping the (resolved) source file path to a set of line numbers.
    """
    source_file = os.path.realpath(source_file)

    # Include our CUDA stubs directory so parsing works without CUDA toolkit
    stubs_dir = os.path.join(os.path.dirname(__file__), 'stubs')

    args = [
        '-x', 'cuda',
        '--cuda-gpu-arch=sm_70',
        '-nocudalib',
        '-nocudainc',
        '-D__CUDACC__',
        '-D__CUDA_ARCH__=700',
        '-std=c++20',
        '-fsyntax-only',
        '-I', stubs_dir,
        *_get_system_include_args(),
    ]

    if include_paths:
        for p in include_paths:
            args.extend(['-I', str(p)])

    if extra_args:
        args.extend(extra_args)

    index = Index.create()
    tu = index.parse(
        source_file,
        args=args,
        options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        | TranslationUnit.PARSE_SKIP_FUNCTION_BODIES * 0,  # We need bodies
    )

    if tu is None:
        raise RuntimeError(f"Failed to parse {source_file}")

    # Collect diagnostics but don't fail — CUDA headers may produce warnings
    errors = [d for d in tu.diagnostics if d.severity >= 3]
    if errors:
        import sys
        for d in errors[:5]:
            print(f"  clang warning: {d}", file=sys.stderr)

    cuda_funcs = _find_cuda_functions(tu.cursor, source_file)

    result: dict[str, set[int]] = {}
    for func in cuda_funcs:
        lines = _collect_executable_lines(func, source_file)
        if lines:
            result.setdefault(source_file, set()).update(lines)

    return result


def instrument_file(
    source_file: str,
    output_dir: str,
    source_root: str,
    counter_start: int = 0,
    include_paths: list[str] | None = None,
    extra_args: list[str] | None = None,
    dual_compilation_patterns: list[str] | None = None,
) -> InstrumentationResult:
    """Instrument a single CUDA source file.

    Writes the instrumented version to output_dir, preserving the relative
    path from source_root. Returns the counter mappings.

    Args:
        dual_compilation_patterns: List of glob patterns for files compiled by
            both the host compiler and NVCC. These files get ``#ifdef`` guards
            around instrumented code. Example: ``["operations.cuh"]``
    """
    source_file = os.path.realpath(source_file)
    source_root = os.path.realpath(source_root)

    executable_lines = find_executable_lines(source_file, include_paths, extra_args)
    lines_to_instrument = sorted(executable_lines.get(source_file, set()))

    # .cu files always get an instrumented copy (they need the runtime include
    # even if they contain no device functions themselves), while .cuh files
    # are skipped when they have no instrumentable lines.
    is_cu_file = source_file.endswith('.cu')
    if not lines_to_instrument and not is_cu_file:
        return InstrumentationResult(next_counter_id=counter_start)

    # Read the original source
    with open(source_file, 'r') as f:
        original_lines = f.readlines()

    # Filter out lines where inserting a statement would break syntax
    # (e.g. 'else if', 'else {', case/default labels, closing braces)
    def _safe_to_instrument(line_no: int) -> bool:
        if line_no < 1 or line_no > len(original_lines):
            return True
        text = original_lines[line_no - 1].lstrip()
        if text.startswith('else') or text.startswith('}'):
            return False
        if text.startswith('case ') or text.startswith('default:'):
            return False
        return True

    lines_to_instrument = [ln for ln in lines_to_instrument if _safe_to_instrument(ln)]

    # Build counter mappings
    result = InstrumentationResult()
    counter_id = counter_start

    # Map line number -> counter ID
    line_to_counter: dict[int, int] = {}
    for line_no in lines_to_instrument:
        mapping = CounterMapping(id=counter_id, file=source_file, line=line_no)
        result.mappings.append(mapping)
        line_to_counter[line_no] = counter_id
        counter_id += 1

    result.next_counter_id = counter_id

    # Determine if this file needs the #ifdef guard (compiled by both host and NVCC)
    needs_ifdef = _file_needs_ifdef_guard(source_file, dual_compilation_patterns)

    # Rewrite the source
    instrumented = []

    # Add the runtime include at the top, after any existing header guard
    include_line = f'#include "{RUNTIME_HEADER}"\n'
    if needs_ifdef:
        include_line = (
            f'#ifdef {COVERAGE_DEFINE}\n'
            f'#include "{RUNTIME_HEADER}"\n'
            '#endif\n'
        )

    # Find the right place to insert the include (after header guard or first #include)
    insert_pos = 0
    for i, line in enumerate(original_lines):
        stripped = line.strip()
        if stripped.startswith('#include') or stripped.startswith('#define') and i > 0:
            insert_pos = i
            break
        if stripped.startswith('#ifndef') and i < 5:
            # Header guard — skip the #define too
            insert_pos = i + 2
            continue

    # Find first #include to insert after
    for i, line in enumerate(original_lines):
        if line.strip().startswith('#include'):
            insert_pos = i
            break

    # Insert after the last consecutive #include block
    for i in range(insert_pos, len(original_lines)):
        if original_lines[i].strip().startswith('#include'):
            insert_pos = i + 1
        elif original_lines[i].strip() == '':
            continue
        else:
            break

    for i, line in enumerate(original_lines):
        line_no = i + 1

        if i == insert_pos:
            instrumented.append('\n')
            instrumented.append(include_line)
            instrumented.append('\n')

        if line_no in line_to_counter:
            cid = line_to_counter[line_no]
            # Detect the indentation of the current line
            indent = _get_indentation(line)
            hit_call = f'{indent}{COUNTER_NAMESPACE}::hit({cid});\n'
            if needs_ifdef:
                hit_call = (
                    f'{indent}#ifdef {COVERAGE_DEFINE}\n'
                    f'{indent}{COUNTER_NAMESPACE}::hit({cid});\n'
                    f'{indent}#endif\n'
                )
            instrumented.append(hit_call)

        instrumented.append(line)

    # Write instrumented file to shadow directory
    rel_path = os.path.relpath(source_file, source_root)
    output_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.writelines(instrumented)

    result.instrumented_files.append(output_path)
    return result


def _file_needs_ifdef_guard(
    source_file: str,
    dual_compilation_patterns: list[str] | None = None,
) -> bool:
    """Check if a file is compiled by both host compiler and NVCC.

    A file needs #ifdef guards around instrumented code if it is included
    in both host-compiled and NVCC-compiled translation units.

    Detection methods (in order):
    1. Explicit match against dual_compilation_patterns
    2. .cu files and files under a cuda/ directory are assumed NVCC-only
    3. Files containing common dual-compilation guards (#ifdef BUILD_WITH_CUDA,
       #ifdef __CUDACC__, etc.) are assumed dual
    """
    # .cu files are NVCC-only translation units
    if source_file.endswith('.cu'):
        return False

    # Check explicit patterns
    if dual_compilation_patterns:
        from fnmatch import fnmatch
        for pattern in dual_compilation_patterns:
            if fnmatch(source_file, pattern) or fnmatch(os.path.basename(source_file), pattern):
                return True

    # Files under a cuda/ directory are typically NVCC-only
    if '/cuda/' in source_file or '\\cuda\\' in source_file:
        return False

    # Heuristic: check for common dual-compilation guards
    try:
        with open(source_file, 'r') as f:
            content = f.read(4096)
        dual_guards = [
            '#ifdef BUILD_WITH_CUDA',
            '#ifndef BUILD_WITH_CUDA',
            '#ifdef __CUDACC__',
            '#ifndef __CUDACC__',
            '#if defined(__CUDACC__)',
            '#if defined(BUILD_WITH_CUDA)',
        ]
        return any(guard in content for guard in dual_guards)
    except OSError:
        return False


def _get_indentation(line: str) -> str:
    """Extract the leading whitespace from a line."""
    return line[:len(line) - len(line.lstrip())]


def instrument_files(
    source_files: list[str],
    output_dir: str,
    source_root: str,
    include_paths: list[str] | None = None,
    extra_args: list[str] | None = None,
    dual_compilation_patterns: list[str] | None = None,
) -> InstrumentationResult:
    """Instrument multiple CUDA source files.

    Writes instrumented files to output_dir and a mapping.json file.

    Args:
        dual_compilation_patterns: Glob patterns for files compiled by both
            host and NVCC. Passed through to instrument_file().
    """
    output_dir = os.path.realpath(output_dir)
    source_root = os.path.realpath(source_root)

    os.makedirs(output_dir, exist_ok=True)

    combined = InstrumentationResult()
    counter_id = 0

    for src in source_files:
        result = instrument_file(
            src, output_dir, source_root,
            counter_start=counter_id,
            include_paths=include_paths,
            extra_args=extra_args,
            dual_compilation_patterns=dual_compilation_patterns,
        )
        combined.mappings.extend(result.mappings)
        combined.instrumented_files.extend(result.instrumented_files)
        counter_id = result.next_counter_id

    combined.next_counter_id = counter_id

    # Copy the runtime header into the shadow directory
    runtime_src = os.path.join(os.path.dirname(__file__), 'runtime.cuh')
    runtime_dst = os.path.join(output_dir, RUNTIME_HEADER)
    shutil.copy2(runtime_src, runtime_dst)

    # Write mapping.json
    mapping_path = os.path.join(output_dir, 'mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(
            {
                "num_counters": combined.next_counter_id,
                "mappings": [m.to_dict() for m in combined.mappings],
            },
            f,
            indent=2,
        )

    print(f"Instrumented {len(combined.instrumented_files)} files, "
          f"{combined.next_counter_id} counters")
    print(f"Mapping written to {mapping_path}")
    print(f"Runtime header copied to {runtime_dst}")

    return combined
