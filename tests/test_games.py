"""Game rules, reproducibility, Markov encoding and optional Gym adapter."""

import pytest

from prometheus_rl.environments.games import HazardGrid, Snake, potential_reward


def test_snake_seed_and_observation():
    a, b = Snake(), Snake()
    assert a.reset(seed=42) == b.reset(seed=42)
    assert a.food not in a.body
    obs = a.observation()
    assert len(obs) == 76 and all(0 <= x <= 1 for x in obs)
    assert sum(obs[36:72]) == 1 and sum(obs[72:]) == 1
    for action in (1, 2, 1):
        assert a.step(action) == b.step(action)


def test_snake_growth_and_heading():
    env = Snake()
    env.reset(seed=1)
    env.food = (3, 3)
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == 1 and not terminated and not truncated
    assert len(env.body) == 4 and info["score"] == 1
    assert env.food not in env.body
    env.food = (5, 5)
    env.step(0)
    assert env.body[0] == (3, 2) and env.heading == 0


def test_snake_tail_vacates_and_body_collision():
    env = Snake()
    env.reset(seed=0)
    env.body = [(1, 1), (1, 2), (2, 2), (2, 1)]
    env.heading = 0
    env.food = (5, 5)
    assert env.step(2)[2] is False
    assert env.body[0] == (2, 1)
    env.body = [(1, 1), (1, 2), (2, 2), (2, 1), (3, 1)]
    env.heading = 0
    assert env.step(2)[1:4] == (-1.0, True, False)


def test_snake_wall_collision_and_lifecycle():
    env = Snake(4)
    with pytest.raises(RuntimeError):
        env.step(1)
    env.reset(seed=0)
    env.food = (0, 0)
    env.step(1)
    assert env.step(1)[1:4] == (-1.0, True, False)
    with pytest.raises(RuntimeError):
        env.step(1)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(3)


def test_snake_full_board_win():
    env = Snake(4)
    env.reset(seed=0)
    env.body = [(2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1),
                (3, 1), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3),
                (1, 3), (2, 3), (3, 3)]
    env.heading = 1
    env.food = (3, 0)
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == 1 and terminated and not truncated
    assert info["reason"] == "filled" and env.food is None
    assert len(env.body) == 16


def test_time_limit_and_terminal_precedence():
    env = Snake(max_steps=1)
    env.reset(seed=0)
    env.food = (5, 5)
    assert env.step(1)[2:4] == (False, True)
    grid = HazardGrid(("SG",), max_steps=1)
    grid.reset()
    assert grid.step(1)[2:4] == (True, False)


def test_body_order_is_observed():
    env = Snake()
    env.reset(seed=0)
    first = env.observation()
    env.body[1], env.body[2] = env.body[2], env.body[1]
    assert env.observation() != first


def test_grid_walls_hazards_and_goal():
    env = HazardGrid(("S#G", ".H."))
    start, _ = env.reset()
    assert env.step(1)[0] == start  # wall
    assert env.step(0)[0] == start  # boundary
    env.step(2)
    assert env.step(1)[1:4] == (-1.0, True, False)
    env = HazardGrid(("S.G",))
    env.reset()
    assert env.step(1)[1] == -0.04
    _, reward, terminated, truncated, info = env.step(1)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["success"]


@pytest.mark.parametrize("layout", [(), ("S",), ("SSG",), ("SG", "."), ("S?G",)])
def test_invalid_map(layout):
    with pytest.raises(ValueError):
        HazardGrid(layout)


def test_shaping_telescopes():
    gamma = 0.9
    base = HazardGrid(("S.G",))
    shaped = HazardGrid(("S.G",), shaping_gamma=gamma)
    base.reset()
    shaped.reset()
    returns = [0.0, 0.0]
    for t in range(2):
        for i, env in enumerate((base, shaped)):
            returns[i] += gamma**t * env.step(1)[1]
    assert returns[1] - returns[0] == pytest.approx(2.0)
    assert potential_reward(1, gamma, -2, 10, terminated=True) == 3


@pytest.mark.parametrize("name,dim,actions", [("snake", 76, 3), ("gridworld", 25, 4)])
def test_gym_contract(name, dim, actions):
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from prometheus_rl.environments.gym_games import GameEnv

    env = GameEnv(name, render_mode="rgb_array")
    try:
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=4)
        assert obs.shape == (dim,) and env.action_space.n == actions
        assert env.observation_space.contains(obs)
        assert env.render().shape == (env.game.height * 48, env.game.width * 48, 3)
    finally:
        env.close()


@pytest.mark.parametrize("name", ["snake", "gridworld"])
def test_online_game_training(name, tmp_path):
    pytest.importorskip("torch")
    from prometheus_rl.agents.dqn import DQNConfig
    from prometheus_rl.environments.games import make_game
    from prometheus_rl.training.cartpole import TrainConfig, run_experiment

    spec = make_game(name)
    config = DQNConfig(observation_dim=spec.observation_dim,
                       action_dim=spec.action_dim, hidden_dims=(16,),
                       capacity=128, batch_size=8, warmup=8,
                       double=True, dueling=True, prioritized=True)
    result = run_experiment(tmp_path / name, TrainConfig(steps=64),
                            eval_episodes=2, plots=False, dqn_config=config,
                            factory=lambda: make_game(name), environment_id=name)
    assert result["updates"] == 15
    assert result["environment"] == name
    assert len(result["greedy"]["returns"]) == 2
