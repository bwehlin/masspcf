"""CLI for GPUCov — GPU kernel coverage tool.

Usage:
    gpucov instrument --source-root . --output-dir build/_gpucov --files FILE...
    gpucov zerocounters --dump "coverage/cuda_*.bin"
    gpucov collect --dump "coverage/cuda_*.bin" --mapping FILE [--lcov FILE] [--summary FILE]

Or as a module:
    python -m gpucov instrument ...
    python -m gpucov zerocounters ...
    python -m gpucov collect ...
"""

import argparse
import sys


def cmd_instrument(args):
    from .instrumenter import instrument_files

    include_paths = args.include_paths or []
    extra_args = args.extra_args or []

    result = instrument_files(
        source_files=args.files,
        output_dir=args.output_dir,
        source_root=args.source_root,
        include_paths=include_paths,
        extra_args=extra_args,
    )

    if result.next_counter_id == 0:
        print("Warning: no executable lines found in device functions", file=sys.stderr)
        return 1

    print(f"Total counters: {result.next_counter_id}")
    return 0


def cmd_collect(args):
    import glob
    from .collector import collect_and_report

    if not args.lcov and not args.summary:
        print("Error: specify at least one of --lcov or --summary", file=sys.stderr)
        return 1

    # Expand glob patterns in dump paths
    dump_paths = []
    for pattern in args.dump:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"Warning: no files matched '{pattern}'", file=sys.stderr)
        dump_paths.extend(expanded)

    if not dump_paths:
        print("Error: no dump files found", file=sys.stderr)
        return 1

    if len(dump_paths) > 1:
        print(f"Merging {len(dump_paths)} dump files", file=sys.stderr)

    coverage = collect_and_report(
        dump_path=dump_paths,
        mapping_path=args.mapping,
        lcov_path=args.lcov,
        summary_path=args.summary,
    )

    if not coverage:
        print("Warning: no coverage data collected", file=sys.stderr)
        return 1

    return 0


def cmd_zerocounters(args):
    import glob
    import os

    patterns = args.dump
    removed = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            os.remove(path)
            removed += 1

    if removed:
        print(f"Removed {removed} dump file{'s' if removed != 1 else ''}")
    else:
        print("No dump files found")

    return 0


def cmd_cmake_dir(args):
    import os
    cmake_dir = os.path.join(os.path.dirname(__file__), 'cmake')
    print(cmake_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='gpucov',
        description='GPUCov — line-level coverage for GPU device code',
    )
    parser.add_argument(
        '--cmake-dir', action='store_true',
        help='Print the path to the GPUCov CMake module directory and exit',
    )
    subparsers = parser.add_subparsers(dest='command')

    # instrument subcommand
    p_inst = subparsers.add_parser(
        'instrument',
        help='Instrument CUDA source files for coverage',
    )
    p_inst.add_argument(
        '--source-root', required=True,
        help='Project root directory (for computing relative paths)',
    )
    p_inst.add_argument(
        '--output-dir', required=True,
        help='Shadow directory for instrumented sources',
    )
    p_inst.add_argument(
        '--files', nargs='+', required=True,
        help='CUDA source files to instrument',
    )
    p_inst.add_argument(
        '-I', '--include-paths', nargs='*', default=[],
        help='Additional include paths for clang',
    )
    p_inst.add_argument(
        '--extra-args', nargs='*', default=[],
        help='Extra arguments passed to clang parser',
    )
    p_inst.set_defaults(func=cmd_instrument)

    # collect subcommand
    p_coll = subparsers.add_parser(
        'collect',
        help='Collect coverage data and produce reports',
    )
    p_coll.add_argument(
        '--dump', nargs='+', required=True,
        help='Path(s) to binary counter dump file(s). Supports globs. '
             'Multiple dumps are merged by summing counters.',
    )
    p_coll.add_argument(
        '--mapping', required=True,
        help='Path to mapping.json from instrument step',
    )
    p_coll.add_argument(
        '--lcov',
        help='Output path for lcov .info file',
    )
    p_coll.add_argument(
        '--summary',
        help='Output path for coverage-summary.json',
    )
    p_coll.set_defaults(func=cmd_collect)

    # zerocounters subcommand
    p_zero = subparsers.add_parser(
        'zerocounters',
        help='Remove dump files from a previous run',
    )
    p_zero.add_argument(
        '--dump', nargs='+', required=True,
        help='Glob pattern(s) matching dump files to remove. '
             'Use the same pattern as GPUCOV_OUTPUT.',
    )
    p_zero.set_defaults(func=cmd_zerocounters)

    args = parser.parse_args()

    if args.cmake_dir:
        sys.exit(cmd_cmake_dir(args))

    if not args.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
