"""Normalize the three experiment formats without pooling incompatible runs."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean, stdev


@dataclass(frozen=True)
class Run:
    path: Path
    environment: str
    algorithm: str
    seed: int
    steps: int
    eval_seed: int
    episodes: int
    score: float
    baseline: float
    protocol: str


def read_run(path: Path) -> Run:
    data = json.loads((path / "summary.json").read_text())
    if not isinstance(data, dict):
        raise ValueError("summary must be a JSON object")
    if data.get("algorithm") == "ppo":
        game = data["game"]
        env = {"cartpole": "CartPole-v1", "pendulum": "Pendulum-v1"}.get(
            game, f"Prometheus-{game}-v0"
        )
        training = {"steps": data["steps"], "seed": data["seed"],
                    "horizon": data["horizon"]}
        config = data["config"]
        greedy, random = data["greedy"]["returns"], data["random"]["returns"]
        algorithm = "ppo"
        eval_seed = data["eval_seed"]
    else:
        env, training = data["environment"], data["training"]
        config = data.get("dqn", data.get("agent"))
        if not isinstance(config, dict):
            raise ValueError("missing agent configuration")
        algorithm = "visual-dqn" if "agent" in data else "dqn"
        for flag, suffix in (("double", "double"), ("dueling", "dueling"),
                             ("prioritized", "per")):
            if config.get(flag):
                algorithm += "-" + suffix
        if "greedy_scores" in data:
            greedy, random = data["greedy_scores"], data["random_scores"]
            eval_seed = data["eval_seed"]
        else:
            greedy, random = data["greedy"]["returns"], data["random"]["returns"]
            eval_seed = data["evaluation_seed"]
    if (not greedy or len(greedy) != len(random)
            or not all(isfinite(x) for x in (*greedy, *random))):
        raise ValueError("invalid evaluation returns")
    if not isinstance(env, str) or not env or not isinstance(config, dict):
        raise ValueError("invalid environment or configuration")
    seed, steps = training["seed"], training["steps"]
    if (type(seed) is not int or type(steps) is not int or steps < 1
            or seed < 0 or type(eval_seed) is not int or eval_seed < 0):
        raise ValueError("invalid experiment budget or seed")
    protocol = json.dumps({
        "training": {k: v for k, v in training.items() if k != "seed"},
        "config": config, "device": data.get("device"),
        "git": data.get("git", data.get("commit")),
        "versions": data.get("versions", {"torch": data.get("torch")}),
        "threads": data.get("torch_threads"),
        "eval_max_steps": data.get("eval_max_steps"),
        "frame_skip": data.get("frame_skip"),
        "reward_clipping": data.get("reward_clipping"),
    }, sort_keys=True)
    return Run(path, env, algorithm, seed, steps, eval_seed, len(greedy),
               mean(greedy), mean(random), protocol)


def discover(root: Path) -> tuple[list[Run], list[str]]:
    runs, errors = [], []
    for file in sorted(root.glob("*/summary.json")):
        try:
            runs.append(read_run(file.parent))
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            errors.append(f"{file.parent.name}: {error}")
    return runs, errors


def aggregate(runs: list[Run]) -> list[dict]:
    groups = {}
    for run in runs:
        key = (run.environment, run.algorithm, run.steps, run.episodes, run.protocol)
        groups.setdefault(key, []).append(run)
    output = []
    for (env, algo, steps, episodes, protocol), group in sorted(groups.items()):
        seeds = [r.seed for r in group]
        duplicate = len(set(seeds)) != len(seeds)
        scores = [r.score for r in group]
        recorded = json.loads(protocol)
        missing = [k for k in ("git", "device", "versions", "threads")
                   if not recorded.get(k)]
        versions = recorded.get("versions")
        if not isinstance(versions, dict) or not any(versions.values()):
            if "versions" not in missing:
                missing.append("versions")
        output.append({
            "environment": env, "algorithm": algo, "steps": steps,
            "eval_episodes": episodes, "seeds": sorted(seeds),
            "eval_seeds": sorted(r.eval_seed for r in group),
            "n": len(group), "duplicate_seeds": duplicate,
            "mean": None if duplicate else mean(scores),
            "sd": stdev(scores) if len(scores) > 1 and not duplicate else None,
            "random_mean": None if duplicate else mean(r.baseline for r in group),
            "protocol": recorded, "missing_metadata": missing,
        })
    return output
