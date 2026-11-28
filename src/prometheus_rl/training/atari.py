"""Online visual DQN experiments; separate seeds and raw-score evaluation."""

import argparse
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np

from prometheus_rl.agents.visual import VisualAgent, VisualConfig
from prometheus_rl.environments.atari import GAMES, make_atari
from prometheus_rl.memory.pixels import PixelExperience
from prometheus_rl.training.cartpole import (
    Episode,
    TrainConfig,
    _commit,
    _versions,
    write_results,
)


def train(agent: VisualAgent, config: TrainConfig, factory: Callable):
    env = factory()
    records = []
    try:
        obs, _ = env.reset(seed=config.seed)
        obs = np.array(obs, copy=True)
        total, length, losses = 0.0, 0, []
        for step in range(1, config.steps + 1):
            epsilon = config.epsilon(step - 1)
            action = agent.act(obs, epsilon=epsilon)
            nxt, reward, terminated, truncated, _ = env.step(action)
            agent.observe(PixelExperience(obs, action, float(np.sign(reward)),
                                          nxt, terminated, truncated))
            if step % config.train_every == 0:
                result = agent.update()
                if result is not None:
                    losses.append(result.loss)
            total += reward
            length += 1
            obs = np.array(nxt, copy=True)
            if terminated or truncated:
                record = Episode(len(records) + 1, step, total, length, epsilon,
                                 mean(losses) if losses else None,
                                 terminated, truncated)
                records.append(record)
                if record.episode % 10 == 0:
                    print(f"step={step} score={total:g} epsilon={epsilon:.3f}",
                          flush=True)
                if step < config.steps:
                    obs, _ = env.reset(seed=None)
                    obs = np.array(obs, copy=True)
                total, length, losses = 0.0, 0, []
    finally:
        env.close()
    return tuple(records)


def evaluate(agent: VisualAgent, factory: Callable, *, episodes: int, seed: int,
             random_policy: bool = False, max_steps: int = 27000):
    if episodes < 1 or seed < 0 or max_steps < 1:
        raise ValueError("invalid evaluation budget")
    env = factory()
    rng = Random(seed)
    scores = []
    try:
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            total = 0.0
            for _ in range(max_steps):
                action = (rng.randrange(agent.config.action_dim) if random_policy
                          else agent.act(obs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            scores.append(total)
    finally:
        env.close()
    return scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="prometheus-rl atari")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--game", choices=GAMES, default="breakout")
    fit.add_argument("--steps", type=int, default=100000)
    fit.add_argument("--capacity", type=int, default=5000)
    fit.add_argument("--warmup", type=int, default=1000)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--eval-seed", type=int, default=1000000)
    fit.add_argument("--eval-episodes", type=int, default=5)
    fit.add_argument("--eval-max-steps", type=int, default=27000)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--vanilla", action="store_true")
    for cmd in ("evaluate", "play"):
        p = sub.add_parser(cmd)
        p.add_argument("model", type=Path)
        p.add_argument("--episodes", type=int, default=3)
        p.add_argument("--seed", type=int, default=1000000)
        p.add_argument("--max-steps", type=int, default=27000)
        p.add_argument("--device", default="cpu")
    a = parser.parse_args(argv)
    if a.command != "train":
        agent, game = VisualAgent.load(a.model, device=a.device)

        def playback_factory():
            return make_atari(game, render_mode="human" if a.command == "play"
                              else None)

        print(evaluate(agent, playback_factory, episodes=a.episodes, seed=a.seed,
                       max_steps=a.max_steps))
        return 0
    cfg = TrainConfig(steps=a.steps, seed=a.seed, decay_steps=100000)
    if (a.eval_episodes < 1 or a.eval_max_steps < 1
            or a.eval_seed < a.seed + a.steps + 1):
        parser.error("invalid evaluation budget or overlapping seed ranges")

    def factory():
        return make_atari(a.game)

    env = factory()
    try:
        actions = int(env.action_space.n)
    finally:
        env.close()
    agent = VisualAgent(VisualConfig(actions, capacity=a.capacity,
                        batch_size=a.batch_size, warmup=a.warmup,
                        double=not a.vanilla, dueling=not a.vanilla),
                        seed=a.seed, device=a.device)
    a.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    records = train(agent, cfg, factory)
    duration = perf_counter() - start
    agent.save(a.output / "model.pt", game=a.game)
    greedy = evaluate(agent, factory, episodes=a.eval_episodes, seed=a.eval_seed,
                      max_steps=a.eval_max_steps)
    random = evaluate(agent, factory, episodes=a.eval_episodes, seed=a.eval_seed,
                      max_steps=a.eval_max_steps, random_policy=True)
    from importlib.metadata import version

    write_results(a.output, records, {
        "environment": GAMES[a.game], "training": asdict(cfg),
        "agent": asdict(agent.config), "device": a.device, "git": _commit(),
        "versions": {**_versions(), "ale-py": version("ale-py")},
        "train_seconds": duration, "optimizer_updates": agent.updates,
        "pixel_bytes": agent.replay.pixel_bytes, "frame_skip": 4,
        "reward_clipping": True, "eval_seed": a.eval_seed,
        "eval_max_steps": a.eval_max_steps, "greedy_scores": greedy,
        "random_scores": random, "greedy_mean": mean(greedy),
        "random_mean": mean(random),
    })
    print(f"Greedy mean={mean(greedy):g}; random mean={mean(random):g}")
    return 0
