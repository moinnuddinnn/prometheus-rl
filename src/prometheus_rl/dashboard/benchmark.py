"""Export independent-seed benchmark statistics without extra dependencies."""

import argparse
import json
from pathlib import Path

from prometheus_rl.dashboard.results import aggregate, discover


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    runs, errors = discover(args.runs)
    report = json.dumps({"groups": aggregate(runs), "errors": errors}, indent=2)
    if args.output:
        with args.output.open("x") as handle:
            handle.write(report + "\n")
    else:
        print(report)
    return 0
