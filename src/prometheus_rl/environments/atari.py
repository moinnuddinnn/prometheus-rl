"""Optional ALE dependency and standard grayscale preprocessing."""

GAMES = {"breakout": "ALE/Breakout-v5", "pong": "ALE/Pong-v5"}


def make_atari(game: str, *, render_mode: str | None = None):
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    gym.register_envs(ale_py)
    env = gym.make(GAMES[game], frameskip=1, repeat_action_probability=0.25,
                   render_mode=render_mode)
    try:
        env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                                 terminal_on_life_loss=False, grayscale_obs=True,
                                 scale_obs=False)
        return FrameStackObservation(env, stack_size=4)
    except Exception:
        env.close()
        raise
