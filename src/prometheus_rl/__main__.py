"""Command-line entry point for Prometheus-RL."""

import argparse
import sys
from collections.abc import Sequence

from prometheus_rl import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Report project status or version."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"cartpole", "snake", "gridworld"}:
        try:
            from prometheus_rl.training.cartpole_cli import main as cartpole_main

            return cartpole_main(argv[1:], game=argv[0])
        except ModuleNotFoundError as error:
            if error.name not in {
                "torch", "gymnasium", "matplotlib", "pygame", "numpy"
            }:
                raise
            raise SystemExit("Install the rl extra: pip install -e '.[rl]'") from error
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Games: cartpole, snake, gridworld (each supports --help)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "command", nargs="?",
        choices=["mdp-demo", "q-demo", "network-demo", "replay-demo",
                 "dqn-demo", "variants-demo", "per-demo"]
    )
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
    elif args.command == "network-demo":
        try:
            from prometheus_rl.networks.demo import run_demo
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            parser.error("install the neural extra: pip install -e '.[neural]'")
        count, initial, final = run_demo()
        print(f"Trainable parameters: {count}")
        print(f"Initial fixed-target Huber loss: {initial:.8f}")
        print(f"Final fixed-target Huber loss: {final:.8f}")
        print("Fixed-target regression only; not DQN training.")
    elif args.command == "replay-demo":
        from prometheus_rl.memory.replay import replay_demo

        retained, sampled, masks = replay_demo()
        print(f"Retained rewards (oldest first): {retained}")
        print(f"Sampled rewards: {sampled}")
        print(f"Bootstrap masks: {masks}")
    elif args.command == "per-demo":
        from prometheus_rl.memory.prioritized import PrioritizedReplayBuffer
        from prometheus_rl.memory.replay import Transition

        replay = PrioritizedReplayBuffer(4, 1, 1, alpha=1.0, seed=7)
        for i in range(4):
            replay.append(Transition((float(i),), 0, 0.0, (0.0,), False))
        replay.update_priorities(range(4), (1.0, 2.0, 4.0, 8.0))
        sample = replay.sample_weighted(4, beta=1.0)
        print(f"Sampled slots: {sample.indices}")
        print(f"Probabilities: {tuple(round(p, 4) for p in sample.probabilities)}")
        print(f"Weights: {tuple(round(w, 4) for w in sample.weights)}")
    elif args.command in {"dqn-demo", "variants-demo"}:
        try:
            from prometheus_rl.agents.dqn_demo import run_demo
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            parser.error("install the neural extra: pip install -e '.[neural]'")
        if args.command == "variants-demo":
            for label, double, dueling in (
                ("vanilla", False, False), ("double", True, False),
                ("dueling", False, True), ("double-dueling", True, True),
            ):
                q, v, action = run_demo(double=double, dueling=dueling)
                print(f"{label}: Q=({q[0]:.4f}, {q[1]:.4f}) "
                      f"wait={v:.4f} action={action}")
            print("Fixed-replay toy MDP; not online CartPole training.")
            return 0
        q, v, action = run_demo()
        print(f"Learned start Q-values: ({q[0]:.4f}, {q[1]:.4f})")
        print(f"Learned wait value: {v:.4f}")
        print(f"Greedy start action: {action}")
        print("Fixed-replay toy MDP; not online CartPole training.")
    else:
        print("Prometheus-RL: Block 09 DQN agent implemented; "
              "CartPole, Snake, GridWorld and prioritized replay available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
