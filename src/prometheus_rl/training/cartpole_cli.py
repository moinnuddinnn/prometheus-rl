"""CLI for training, evaluation and human-rendered checkpoint playback."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from prometheus_rl.agents.dqn import DQNAgent, DQNConfig
from prometheus_rl.environments.games import make_game
from prometheus_rl.training.cartpole import (
    TrainConfig,
    evaluate,
    make_cartpole,
    run_experiment,
)


def main(argv: Sequence[str] | None = None, *, game: str = "cartpole") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--double", action="store_true")
    train_parser.add_argument("--dueling", action="store_true")
    train_parser.add_argument("--prioritized", action="store_true")
    train_parser.add_argument("--per-alpha", type=float, default=0.6)
    train_parser.add_argument("--per-beta-start", type=float, default=0.4)
    train_parser.add_argument("--per-beta-steps", type=int, default=25_000)
    train_parser.add_argument("--steps", type=int, default=100_000)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--eval-seed", type=int, default=1_000_000)
    train_parser.add_argument("--eval-episodes", type=int, default=100)
    train_parser.add_argument("--decay-steps", type=int, default=30_000)
    train_parser.add_argument("--train-every", type=int, default=4)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--no-plots", action="store_true")
    for name in ("evaluate", "play"):
        cmd = sub.add_parser(name)
        cmd.add_argument("checkpoint", type=Path)
        cmd.add_argument("--seed", type=int, default=2_000_000)
        cmd.add_argument(
            "--episodes", type=int, default=100 if name == "evaluate" else 3
        )
    for cmd in (train_parser, *[sub.choices[n] for n in ("evaluate", "play")]):
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if game == "cartpole":
        factory = make_cartpole
        observation_dim, action_dim = 4, 2
        environment_id = "CartPole-v1"
    else:
        from prometheus_rl.environments.gym_games import GameEnv

        spec = make_game(game)
        observation_dim, action_dim = spec.observation_dim, spec.action_dim
        environment_id = f"Prometheus-{game}-v0"

        def factory(*, render_mode=None):
            return GameEnv(game, render_mode=render_mode)

    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "train":
        config = TrainConfig(steps=args.steps, decay_steps=args.decay_steps,
                             train_every=args.train_every, seed=args.seed)
        summary = run_experiment(args.output, config, eval_episodes=args.eval_episodes,
                                 eval_seed=args.eval_seed, device=args.device,
                                 plots=not args.no_plots,
                                 factory=factory, environment_id=environment_id,
                                 dqn_config=DQNConfig(
                                     observation_dim=observation_dim,
                                     action_dim=action_dim,
                                     double=args.double, dueling=args.dueling,
                                     prioritized=args.prioritized,
                                     per_alpha=args.per_alpha,
                                     per_beta_start=args.per_beta_start,
                                     per_beta_steps=args.per_beta_steps,
                                 ))
        print(f"Greedy mean: {summary['greedy']['mean']:.2f}")
        print(f"Random mean: {summary['random']['mean']:.2f}")
        print(f"Saved: {args.output}")
    else:
        agent = DQNAgent.load(args.checkpoint, device=args.device)
        if (agent.config.observation_dim, agent.config.action_dim) != (
            observation_dim, action_dim
        ):
            parser.error("checkpoint dimensions do not match the selected game")
        if args.command == "play":
            result = evaluate(agent, episodes=args.episodes, seed=args.seed,
                              factory=lambda: factory(render_mode="human"))
        else:
            result = evaluate(agent, episodes=args.episodes, seed=args.seed,
                              factory=factory)
        print(f"Greedy mean: {result.average:.2f}; std: {result.std:.2f}")
    return 0
