"""A deterministic 4x4 GridWorld; actions N, E, S, W at nonterminal cells."""
from prometheus_rl.core import FiniteMDP, Outcome


def gridworld() -> FiniteMDP:
    """Start at 0, goal at 15, reward +1 entering goal, otherwise -0.05."""
    rows = []
    for state in range(16):
        if state == 15:
            rows.append(())
            continue
        r, c = divmod(state, 4)
        actions = []
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            nxt = nr * 4 + nc if 0 <= nr < 4 and 0 <= nc < 4 else state
            actions.append((Outcome(1.0, nxt, 1.0 if nxt == 15 else -0.05),))
        rows.append(tuple(actions))
    return FiniteMDP(tuple(rows), gamma=0.9, initial=(1.0,) + (0.0,) * 15)
