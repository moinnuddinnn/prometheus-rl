"""Command-line entry point for Prometheus-RL."""

import argparse
from collections.abc import Sequence

from prometheus_rl import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Report project status or version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=["mdp-demo", "q-demo"])
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
    elif args.command == "q-demo":
        from statistics import mean

        from prometheus_rl.core import value_iteration
        from prometheus_rl.environments.gridworld import gridworld
        from prometheus_rl.training.tabular import evaluate, train

        mdp = gridworld()
        print(f"Optimal discounted return: {value_iteration(mdp).values[0]:.6f}")
        for seed in range(5):
            result = train(mdp, seed=seed)
            greedy = evaluate(mdp, result.agent, seed=10000 + seed * 2)
            random = evaluate(mdp, result.agent, seed=10000 + seed * 2,
                              random_policy=True)
            greedy_return = mean(e.discounted_return for e in greedy)
            print(f"Seed {seed}: greedy={greedy_return:.6f} "
                  f"random={mean(e.discounted_return for e in random):.6f} "
                  f"steps={mean(e.steps for e in greedy):.1f} "
                  f"success={mean(e.terminated for e in greedy):.0%}")
    else:
        print("Prometheus-RL: Block 02 tabular Q-learning implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
