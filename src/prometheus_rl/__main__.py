"""Command-line entry point for Prometheus-RL."""

import argparse
from collections.abc import Sequence

from prometheus_rl import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Report project status or version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=["mdp-demo"])
    args = parser.parse_args(argv)
    if args.command == "mdp-demo":
        from prometheus_rl.core import value_iteration
        from prometheus_rl.core.examples import delayed_reward

        mdp = delayed_reward()
        result = value_iteration(mdp)
        print(f"Optimal values: {result.values}")
        print(f"Greedy policy: {result.policy}")
        initial_return = mdp.expected_initial_value(result.values)
        print(f"Initial expected return: {initial_return:.2f}")
        print(f"Bellman residual: {result.residual:.3g}")
    else:
        print("Prometheus-RL: Block 01 MDP planning. No agents implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
