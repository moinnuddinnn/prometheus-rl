"""Small deterministic two-stage MDP with an analytic action-value solution."""

import torch

from prometheus_rl.agents.dqn import DQNAgent, DQNConfig
from prometheus_rl.memory import Transition


def run_demo() -> tuple[tuple[float, float], float, int]:
    """Train from a fixed, fully covered replay dataset (not online CartPole)."""
    cfg = DQNConfig(observation_dim=2, hidden_dims=(16,), gamma=0.9,
                    learning_rate=0.01, capacity=4, batch_size=4, warmup=4,
                    target_interval=10)
    agent = DQNAgent(cfg, seed=7)
    start, wait, end = (1., 0.), (0., 1.), (0., 0.)
    for transition in (
        Transition(start, 0, 2., end, True),
        Transition(start, 1, 0., wait, False),
        Transition(wait, 0, 5., end, True),
        Transition(wait, 1, 1., end, True),
    ):
        agent.observe(transition)
    for _ in range(400):
        agent.update()
    with torch.no_grad():
        q = agent.online(torch.tensor(start)).tolist()
        v = agent.online(torch.tensor(wait)).max().item()
    return (q[0], q[1]), v, agent.act(start)
