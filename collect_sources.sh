#!/bin/bash

# Usage: ./collect_sources.sh [root_dir] [output_file]
# Defaults: root_dir=. output_file=collected_sources.txt

ROOT="${1:-.}"
OUTPUT="${2:-collected_sources.txt}"

EXTENSIONS=("cpp" "h" "cu" "tpp" "py" "cmake" "yml" "yaml")

# Build the -name pattern for find
FIND_ARGS=()
for ext in "${EXTENSIONS[@]}"; do
  FIND_ARGS+=(-o -name "*.${ext}")
done
# Remove leading -o
FIND_ARGS=("${FIND_ARGS[@]:1}")

> "$OUTPUT"

while IFS= read -r -d '' file; do
  echo "========================================" >> "$OUTPUT"
  echo "FILE: $file"                              >> "$OUTPUT"
  echo "========================================" >> "$OUTPUT"
  cat "$file"                                     >> "$OUTPUT"
  echo ""                                         >> "$OUTPUT"
  echo "FILE: $file"
done < <(find "$ROOT" \( "${FIND_ARGS[@]}" \) -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/3rd/*" -not -path "*/benchmarking/*" -not -path "*/examples/*" -not -path "*/docs/*" -not -path "*/cmake-build*" -print0 | sort -z)

echo "Collected into $OUTPUT"
wc -l "$OUTPUT"