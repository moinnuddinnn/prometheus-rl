"""Command-line entry point; algorithm commands arrive in later blocks."""

import argparse
from collections.abc import Sequence

from prometheus_rl import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Report project status or version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    print("Prometheus-RL: Phase 0 initialized. No agents implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
