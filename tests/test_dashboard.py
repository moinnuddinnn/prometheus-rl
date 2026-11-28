import json

import pytest

from prometheus_rl.dashboard.commands import play_command, train_command
from prometheus_rl.dashboard.results import aggregate, discover, read_run
from prometheus_rl.dashboard.server import Dashboard, chart


def sample(tmp_path, name, seed=0, steps=5000, score=4):
    path = tmp_path / name
    path.mkdir()
    data = {
        "environment": "CartPole-v1", "training": {"seed": seed, "steps": steps},
        "dqn": {"double": True}, "evaluation_seed": 1000000 + seed,
        "greedy": {"returns": [score, score]}, "random": {"returns": [1, 3]},
        "device": "cpu", "commit": "abc",
    }
    (path / "summary.json").write_text(json.dumps(data))
    return path


def test_aggregate_independent_seeds(tmp_path):
    sample(tmp_path, "one", score=4)
    sample(tmp_path, "two", seed=1, score=8)
    sample(tmp_path, "other", seed=2, steps=6000)
    runs, errors = discover(tmp_path)
    assert not errors
    groups = aggregate(runs)
    assert len(groups) == 2
    assert groups[0]["mean"] == 6
    assert groups[0]["sd"] == pytest.approx(8**0.5)
    assert groups[0]["random_mean"] == 2
    assert groups[1]["sd"] is None


def test_duplicate_seed(tmp_path):
    a = sample(tmp_path, "a")
    b = sample(tmp_path, "b")
    group = aggregate([read_run(a), read_run(b)])[0]
    assert group["duplicate_seeds"]
    assert group["mean"] is None


def test_malformed_summary(tmp_path):
    path = sample(tmp_path, "broken")
    (path / "summary.json").write_text("{}")
    runs, errors = discover(tmp_path)
    assert not runs and len(errors) == 1


@pytest.mark.parametrize("algorithm,game,flag", [
    ("double-dueling-per", "snake", "--prioritized"),
    ("ppo", "pendulum", "--horizon-not-required"),
    ("visual-dqn", "pong", "--vanilla"),
])
def test_training_command(tmp_path, algorithm, game, flag):
    cmd = train_command(algorithm, game, tmp_path / "run", steps=5000, seed=1,
                        episodes=3, threads=2, device="cpu")
    assert game in cmd and cmd[-1] == str(tmp_path / "run")
    if flag != "--horizon-not-required":
        assert flag in cmd
    if game == "pong":
        assert "--capacity" in cmd and "--eval-max-steps" in cmd


def test_invalid_command(tmp_path):
    with pytest.raises(ValueError, match="incompatible"):
        train_command("ppo", "pong", tmp_path, steps=5000, seed=0,
                      episodes=3, threads=1, device="cpu")


def test_ppo_and_atari_formats(tmp_path):
    for name, data in [
        ("ppo", {"algorithm": "ppo", "game": "pendulum", "config": {},
                 "steps": 5, "seed": 0, "horizon": 2, "eval_seed": 100,
                 "greedy": {"returns": [-2]}, "random": {"returns": [-5]}}),
        ("atari", {"environment": "ALE/Pong-v5", "agent": {"double": True},
                   "training": {"steps": 5, "seed": 0}, "eval_seed": 100,
                   "greedy_scores": [-1], "random_scores": [-3]}),
    ]:
        path = tmp_path / name
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(data))
    runs, errors = discover(tmp_path)
    assert not errors
    assert {r.environment for r in runs} == {"Pendulum-v1", "ALE/Pong-v5"}
    assert {r.algorithm for r in runs} == {"ppo", "visual-dqn-double"}


def test_dashboard_page_and_token(tmp_path):
    path = sample(tmp_path, "<run>")
    (path / "episodes.csv").write_text("step,reward\n1,2\n2,3\n")
    app = Dashboard(tmp_path)
    page = app.page(selected="<run>")
    assert "&lt;run&gt;" in page and "<run>" not in page
    assert "<polyline" in page
    with pytest.raises(ValueError, match="token"):
        app.submit({"action": "stop", "token": "wrong"})
    assert "No data" in chart(path / "missing.csv", "reward")


def test_play_command(tmp_path):
    assert "atari" in play_command("visual-dqn-double", "pong", tmp_path)
