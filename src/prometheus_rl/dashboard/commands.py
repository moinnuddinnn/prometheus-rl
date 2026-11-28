"""Build validated subprocess argument vectors, never shell strings."""

import sys
from pathlib import Path

DQN_GAMES = ("cartpole", "snake", "gridworld")
PPO_GAMES = (*DQN_GAMES, "pendulum")
ATARI_GAMES = ("breakout", "pong")
ALGORITHMS = ("dqn", "double", "dueling", "double-dueling", "per",
              "double-per", "dueling-per", "double-dueling-per", "ppo",
              "visual-dqn", "visual-double-dueling")


def train_command(algorithm: str, game: str, output: Path, *, steps: int,
                  seed: int, episodes: int, threads: int, device: str) -> list[str]:
    if (algorithm not in ALGORITHMS or device not in {"cpu", "cuda"}
            or not 1 <= steps <= 10_000_000 or not 0 <= seed < 2**31
            or not 1 <= episodes <= 1000 or not 1 <= threads <= 32):
        raise ValueError("invalid training parameters")
    prefix = [sys.executable, "-m", "prometheus_rl"]
    if algorithm == "ppo" and game in PPO_GAMES:
        cmd = ["ppo", "train", "--game", game, "--threads", str(threads)]
    elif algorithm.startswith("visual-") and game in ATARI_GAMES:
        cmd = ["atari", "train", "--game", game, "--eval-max-steps", "1000"]
        if algorithm == "visual-dqn":
            cmd.append("--vanilla")
        if steps <= 5000:
            cmd += ["--capacity", "2000", "--batch-size", "16", "--warmup", "500"]
    elif algorithm not in {"ppo", "visual-dqn", "visual-double-dueling"} and (
        game in DQN_GAMES
    ):
        cmd = [game, "train", "--threads", str(threads)]
        for part, flag in (("double", "--double"), ("dueling", "--dueling"),
                           ("per", "--prioritized")):
            if part in algorithm.split("-"):
                cmd.append(flag)
        cmd += ["--decay-steps", str(max(1, steps // 3))]
    else:
        raise ValueError("algorithm is incompatible with environment")
    return prefix + cmd + [
        "--steps", str(steps), "--seed", str(seed), "--eval-episodes", str(episodes),
        "--eval-seed", str(seed + steps + 1_000_000), "--device", device,
        "--output", str(output),
    ]


def play_command(algorithm: str, game: str, checkpoint: Path) -> list[str]:
    if algorithm == "ppo" and game in PPO_GAMES:
        command = "ppo"
    elif algorithm.startswith("visual-") and game in ATARI_GAMES:
        command = "atari"
    elif algorithm.startswith("dqn") and game in DQN_GAMES:
        command = game
    else:
        raise ValueError("unsupported playback environment")
    return [sys.executable, "-m", "prometheus_rl", command, "play",
            str(checkpoint), "--episodes", "1"]
