#!/usr/bin/env python3
"""
collect_sources.py - Collect source files into a single text file.

Usage:
    python collect_sources.py [options] [sources...]

Options:
    -o, --output <file>     Output file (default: collected_sources.txt)
    -e, --ext <ext>         Add extra extension (can be repeated)

Examples:
    python collect_sources.py
    python collect_sources.py src/ include/
    python collect_sources.py src/ myfile.py
    python collect_sources.py -o out.txt src/ include/
    python collect_sources.py -e ts -e js src/
"""

#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import sys
from pathlib import Path

DEFAULT_EXTENSIONS = {"cpp", "h", "cu", "tpp", "py", "cmake", "yml", "yaml"}

EXCLUDE_DIRS = {
    "build",
    ".git",
    "3rd",
    "benchmarking",
    "examples",
    "docs",
}
EXCLUDE_PREFIXES = {"cmake-build"}


def is_excluded(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if any(part.startswith(p) for p in EXCLUDE_PREFIXES):
            return True
    return False


def collect_files(sources: list[Path], extensions: set[str]) -> list[Path]:
    collected = []
    seen = set()

    for source in sources:
        if source.is_file():
            if source.resolve() not in seen:
                seen.add(source.resolve())
                collected.append(source)

        elif source.is_dir():
            matches = sorted(
                p
                for p in source.rglob("*")
                if p.is_file()
                and p.suffix.lstrip(".").lower() in extensions
                and not is_excluded(p.relative_to(source))
            )
            for p in matches:
                if p.resolve() not in seen:
                    seen.add(p.resolve())
                    collected.append(p)

        else:
            print(
                f"Warning: '{source}' is not a file or directory, skipping.",
                file=sys.stderr,
            )

    return collected


def write_output(files: list[Path], output: Path) -> None:
    with output.open("w", encoding="utf-8") as out:
        for file in files:
            out.write("========================================\n")
            out.write(f"FILE: {file}\n")
            out.write("========================================\n")
            try:
                out.write(file.read_text(encoding="utf-8", errors="replace"))
            except OSError as e:
                print(f"Warning: could not read '{file}': {e}", file=sys.stderr)
            out.write("\n")
            print(f"FILE: {file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect source files into a single text file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sources", nargs="*", default=["."], help="Files or directories to collect from"
    )
    parser.add_argument(
        "-o", "--output", default="collected_sources.txt", help="Output file"
    )
    parser.add_argument(
        "-e",
        "--ext",
        action="append",
        default=[],
        help="Extra file extension to include",
    )
    args = parser.parse_args()

    extensions = DEFAULT_EXTENSIONS | {e.lstrip(".").lower() for e in args.ext}
    sources = [Path(s) for s in args.sources]
    output = Path(args.output)

    files = collect_files(sources, extensions)
    write_output(files, output)

    print(
        f"\nCollected {len(files)} file(s) into {output} ({output.stat().st_size:,} bytes)"
    )


if __name__ == "__main__":
    main()
