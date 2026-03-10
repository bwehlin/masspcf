#!/usr/bin/env python3
import json, sys, pathlib

def main(src, dst):
    with open(src) as f:
        data = json.load(f)
    pct = 0.0
    try:
        pct = float(data['data'][0]['totals']['lines']['percent'])
    except Exception:
        pass
    out = {'line_percent': round(pct, 1)}
    pathlib.Path(dst).write_text(json.dumps(out))
    print(f"Wrote {dst} (line_percent={out['line_percent']})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <llvm-cov-summary.json> <coverage-summary.json>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
