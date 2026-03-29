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


@dataclass
class CounterMapping:
    """Maps a counter ID to a source file and line number."""
    id: int
    file: str
    line: int

    def to_dict(self) -> dict:
        return {"id": self.id, "file": self.file, "line": self.line}


@dataclass
class FunctionMapping:
    """Maps a CUDA function to its source location and entry counter."""
    name: str
    file: str
    line: int
    entry_counter_id: int

    def to_dict(self) -> dict:
        return {
            "name": self.name, "file": self.file,
            "line": self.line, "entry_counter_id": self.entry_counter_id,
        }


@dataclass
class BranchMapping:
    """Maps a branch point to its true/false counter IDs."""
    true_counter_id: int
    false_counter_id: int
    file: str
    line: int
    block_id: int

    def to_dict(self) -> dict:
        return {
            "true_counter_id": self.true_counter_id,
            "false_counter_id": self.false_counter_id,
            "file": self.file, "line": self.line,
            "block_id": self.block_id,
        }


@dataclass
class InstrumentationResult:
    """Result of instrumenting a set of source files."""
    mappings: list[CounterMapping] = field(default_factory=list)
    functions: list[FunctionMapping] = field(default_factory=list)
    branches: list[BranchMapping] = field(default_factory=list)
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


_BRANCH_KINDS = {CursorKind.IF_STMT, CursorKind.WHILE_STMT, CursorKind.FOR_STMT}


def _collect_executable_lines(node, source_file: str) -> tuple[set[int], set[int]]:
    """Collect line numbers of executable statements within a function body.

    Returns ``(executable_lines, branch_lines)`` where *branch_lines* is the
    subset of lines that are ``if`` / ``while`` / ``for`` condition headers.
    """
    lines = set()
    branch_lines = set()

    def _walk(n):
        if n.location.file and os.path.realpath(n.location.file.name) == source_file:
            if n.kind in _EXECUTABLE_KINDS:
                lines.add(n.location.line)
            if n.kind in _BRANCH_KINDS:
                branch_lines.add(n.location.line)

        for child in n.get_children():
            _walk(child)

    # Find the compound statement (function body)
    for child in node.get_children():
        if child.kind == CursorKind.COMPOUND_STMT:
            _walk(child)
            break

    return lines, branch_lines


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


def _get_cuda_wrapper_dir(cuda_include: str) -> str:
    """Create a temp directory with a wrapper ``cuda_runtime.h``.

    The wrapper ``#include``s the real header by absolute path (so all
    CUDA types and APIs are available to libclang), then ``#undef``s and
    redefines the CUDA qualifiers as empty.  This avoids the overload
    conflicts that libclang cannot resolve (e.g. ``__host__ __device__
    void memset(...)`` vs the libc ``void memset(...)``).
    """
    import tempfile

    real_header = os.path.join(cuda_include, 'cuda_runtime.h')
    # Use a stable path so we don't litter tmp on repeated runs
    wrapper_dir = os.path.join(tempfile.gettempdir(), 'gpucov_cuda_wrapper')
    os.makedirs(wrapper_dir, exist_ok=True)

    wrapper_path = os.path.join(wrapper_dir, 'cuda_runtime.h')
    content = f"""\
/* Auto-generated by gpucov — do not edit. */
#ifndef GPUCOV_CUDA_RUNTIME_WRAPPER_H
#define GPUCOV_CUDA_RUNTIME_WRAPPER_H

/* Pre-define CUDA qualifiers as empty *before* including the real header.
   The CUDA headers guard their own definitions with:
       #if defined(__CUDACC__) || !defined(__host__)
   Since we define __host__ here (as empty), the guard's !defined(__host__)
   is false, so the header won't redefine it — and libclang avoids the
   __host__ __device__ overload conflicts it can't resolve. */
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__
#define __constant__
#define __managed__

/* Now pull in the real CUDA runtime — all types, enums, and API
   functions become available, with the qualifiers harmlessly blanked. */
#include "{real_header}"

#endif
"""
    with open(wrapper_path, 'w') as f:
        f.write(content)

    return wrapper_dir


def find_executable_lines(
    source_file: str,
    include_paths: list[str] | None = None,
    extra_args: list[str] | None = None,
    cuda_include: str | None = None,
) -> dict[str, set[int]]:
    """Parse a CUDA source file and return executable lines in device functions.

    Returns a dict mapping the (resolved) source file path to a set of line numbers.

    If *cuda_include* is given it should point at the CUDA toolkit include
    directory (e.g. ``/usr/local/cuda/include``).  The real headers will be
    used instead of the minimal built-in stubs, which avoids type-resolution
    errors for projects that use the full CUDA runtime API.
    """
    source_file = os.path.realpath(source_file)

    args = [
        '-x', 'cuda',
        '--cuda-gpu-arch=sm_70',
        '-nocudalib',
        '-D__CUDA_ARCH__=700',
        '-std=c++20',
        '-fsyntax-only',
    ]

    if cuda_include:
        # Use the real CUDA toolkit headers for full type/API coverage.
        # We generate a wrapper cuda_runtime.h that pre-defines CUDA
        # qualifiers as empty, then includes the real header.  The CUDA
        # headers guard their own qualifier definitions with:
        #   #if defined(__CUDACC__) || !defined(__host__)
        # So we must NOT define __CUDACC__ — otherwise the guard's first
        # branch fires and the header forcibly redefines __host__ with its
        # own __location__() built-in that libclang doesn't understand.
        # Without __CUDACC__, the second branch (!defined) checks our
        # pre-existing #define and leaves it alone.
        wrapper_dir = _get_cuda_wrapper_dir(cuda_include)
        args.extend(['-nocudainc', '-I', wrapper_dir, '-I', cuda_include])
    else:
        # Fall back to minimal stubs when no toolkit is available
        stubs_dir = os.path.join(os.path.dirname(__file__), 'stubs')
        args.extend(['-nocudainc', '-D__CUDACC__', '-I', stubs_dir])

    args.extend(_get_system_include_args())

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

    all_lines: set[int] = set()
    all_branch_lines: set[int] = set()
    func_info: list[tuple[str, int, int, int]] = []  # (name, line, extent_start, extent_end)

    for func in cuda_funcs:
        lines, branch_lines = _collect_executable_lines(func, source_file)
        all_lines.update(lines)
        all_branch_lines.update(branch_lines)
        func_info.append((
            func.spelling,
            func.location.line,
            func.extent.start.line,
            func.extent.end.line,
        ))

    result: dict[str, set[int]] = {}
    if all_lines:
        result[source_file] = all_lines

    return result, all_branch_lines, func_info


def instrument_file(
    source_file: str,
    output_dir: str,
    source_root: str,
    counter_start: int = 0,
    include_paths: list[str] | None = None,
    extra_args: list[str] | None = None,
    cuda_include: str | None = None,
) -> InstrumentationResult:
    """Instrument a single CUDA source file.

    Writes the instrumented version to output_dir, preserving the relative
    path from source_root. Returns the counter mappings.

    All injected code is guarded with ``#ifdef __CUDACC__`` so that files
    compiled by both the host compiler and NVCC work without issues.
    """
    source_file = os.path.realpath(source_file)
    source_root = os.path.realpath(source_root)

    executable_lines, branch_lines, func_info = find_executable_lines(
        source_file, include_paths, extra_args, cuda_include)
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
    # (e.g. 'else if', 'else {', case/default labels, closing braces,
    # or lines inside a multi-line function call / initializer).

    # Pre-compute, for each line, whether it sits inside a function-call
    # argument list (paren_depth > 0) but NOT inside a lambda/block body
    # opened within those parens (brace_depth > brace_depth_at_open).
    #
    # _in_call_arg[i] is True for line i (0-based) when a GPUCOV_HIT()
    # statement cannot be inserted because it would land between call args.
    _in_call_arg = [False] * len(original_lines)
    _paren_brace_stack: list[int] = []  # brace depth when each '(' was opened
    pd = 0
    bd = 0
    for i, line in enumerate(original_lines):
        # State at the START of this line decides its status.
        if pd > 0 and _paren_brace_stack and bd <= _paren_brace_stack[-1]:
            _in_call_arg[i] = True
        for ch in line:
            if ch == '(':
                _paren_brace_stack.append(bd)
                pd += 1
            elif ch == ')' and pd > 0:
                pd -= 1
                if _paren_brace_stack:
                    _paren_brace_stack.pop()
            elif ch == '{':
                bd += 1
            elif ch == '}' and bd > 0:
                bd -= 1

    def _is_else_if(line_no: int) -> bool:
        """True when the line is an ``else if (...)`` branch."""
        if line_no < 1 or line_no > len(original_lines):
            return False
        text = original_lines[line_no - 1].lstrip()
        return text.startswith('else if') or text.startswith('else  if')

    def _safe_to_instrument(line_no: int) -> bool:
        if line_no < 1 or line_no > len(original_lines):
            return True
        text = original_lines[line_no - 1].lstrip()
        # else-if is handled specially (condition rewrite), so allow it
        if text.startswith('else if'):
            return True
        if text.startswith('else') or text.startswith('}'):
            return False
        if text.startswith('case ') or text.startswith('default:'):
            return False
        # Skip lines that are plain function-call arguments (but allow
        # lines inside lambda bodies even though they're within parens).
        if _in_call_arg[line_no - 1]:
            return False
        return True

    lines_to_instrument = [ln for ln in lines_to_instrument if _safe_to_instrument(ln)]

    # Determine which instrumentable lines are branch conditions.
    branch_line_set = branch_lines & set(lines_to_instrument)
    # Also treat else-if lines as branches.
    for ln in lines_to_instrument:
        if _is_else_if(ln):
            branch_line_set.add(ln)

    # Build counter mappings.
    # Regular lines get 1 counter (GPUCOV_HIT).
    # Branch lines get 1 line counter + 2 branch counters (GPUCOV_BRANCH).
    result = InstrumentationResult()
    counter_id = counter_start

    line_to_counter: dict[int, int] = {}
    # branch line → (true_counter, false_counter)
    line_to_branch: dict[int, tuple[int, int]] = {}
    branch_block_id = 0

    for line_no in lines_to_instrument:
        # Line counter
        mapping = CounterMapping(id=counter_id, file=source_file, line=line_no)
        result.mappings.append(mapping)
        line_to_counter[line_no] = counter_id
        counter_id += 1

        # Branch counters
        if line_no in branch_line_set:
            tc, fc = counter_id, counter_id + 1
            line_to_branch[line_no] = (tc, fc)
            result.branches.append(BranchMapping(
                true_counter_id=tc, false_counter_id=fc,
                file=source_file, line=line_no, block_id=branch_block_id,
            ))
            counter_id += 2
            branch_block_id += 1

    # Build function mappings: find the first line counter in each function.
    for fname, fline, ext_start, ext_end in func_info:
        entry_cid = None
        for ln in lines_to_instrument:
            if ext_start <= ln <= ext_end and ln in line_to_counter:
                entry_cid = line_to_counter[ln]
                break
        if entry_cid is not None:
            result.functions.append(FunctionMapping(
                name=fname, file=source_file, line=fline,
                entry_counter_id=entry_cid,
            ))

    result.next_counter_id = counter_id

    # --- Rewrite the source ---
    instrumented = []
    include_line = f'#include "{RUNTIME_HEADER}"\n'

    # Find the right place to insert the include (after header guard or first #include)
    insert_pos = 0
    for i, line in enumerate(original_lines):
        stripped = line.strip()
        if stripped.startswith('#include') or stripped.startswith('#define') and i > 0:
            insert_pos = i
            break
        if stripped.startswith('#ifndef') and i < 5:
            insert_pos = i + 2
            continue

    for i, line in enumerate(original_lines):
        if line.strip().startswith('#include'):
            insert_pos = i
            break

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
            branch = line_to_branch.get(line_no)

            if _is_else_if(line_no):
                # Rewrite: else if (cond) → else if (GPUCOV_HIT(N) && GPUCOV_BRANCH(T, F, (cond)))
                m = re.match(r'^(\s*else\s+if\s*)\((.*)$', line, re.DOTALL)
                if m and branch:
                    tc, fc = branch
                    line = f'{m.group(1)}(GPUCOV_HIT({cid}) && GPUCOV_BRANCH({tc}, {fc}, ({m.group(2)}'
                    rp = line.rfind(')')
                    if rp >= 0:
                        line = line[:rp] + '))' + line[rp:]
                elif m:
                    # No branch counters — just line hit
                    line = f'{m.group(1)}(GPUCOV_HIT({cid}) && ({m.group(2)}'
                    rp = line.rfind(')')
                    if rp >= 0:
                        line = line[:rp] + ')' + line[rp:]

            elif branch:
                # Standalone if/while/for: insert GPUCOV_HIT before,
                # and rewrite condition with GPUCOV_BRANCH.
                indent = _get_indentation(line)
                text = line.lstrip()
                tc, fc = branch

                # Match: if (...) / while (...) / for (
                m = re.match(r'^(if\s*|while\s*|for\s*)\((.*)$', text, re.DOTALL)
                if m and m.group(1).strip() != 'for':
                    # Wrap condition: if (cond) → if (GPUCOV_BRANCH(T, F, (cond)))
                    instrumented.append(f'{indent}GPUCOV_HIT({cid});\n')
                    line = f'{indent}{m.group(1)}(GPUCOV_BRANCH({tc}, {fc}, ({m.group(2)}'
                    rp = line.rfind(')')
                    if rp >= 0:
                        line = line[:rp] + '))' + line[rp:]
                else:
                    # for-loops or unrecognized — just line hit, skip branch
                    instrumented.append(f'{indent}GPUCOV_HIT({cid});\n')
            else:
                indent = _get_indentation(line)
                instrumented.append(f'{indent}GPUCOV_HIT({cid});\n')

        instrumented.append(line)

    # Write instrumented file to shadow directory
    rel_path = os.path.relpath(source_file, source_root)
    output_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.writelines(instrumented)

    result.instrumented_files.append(output_path)
    return result


def _get_indentation(line: str) -> str:
    """Extract the leading whitespace from a line."""
    return line[:len(line) - len(line.lstrip())]


def instrument_files(
    source_files: list[str],
    output_dir: str,
    source_root: str,
    include_paths: list[str] | None = None,
    extra_args: list[str] | None = None,
    cuda_include: str | None = None,
) -> InstrumentationResult:
    """Instrument multiple CUDA source files.

    Writes instrumented files to output_dir and a mapping.json file.
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
            cuda_include=cuda_include,
        )
        combined.mappings.extend(result.mappings)
        combined.functions.extend(result.functions)
        combined.branches.extend(result.branches)
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
                "functions": [fn.to_dict() for fn in combined.functions],
                "branches": [br.to_dict() for br in combined.branches],
            },
            f,
            indent=2,
        )

    print(f"Instrumented {len(combined.instrumented_files)} files, "
          f"{combined.next_counter_id} counters")
    print(f"Mapping written to {mapping_path}")
    print(f"Runtime header copied to {runtime_dst}")

    return combined
