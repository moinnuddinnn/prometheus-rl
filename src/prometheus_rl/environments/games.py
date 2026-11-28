"""Dependency-free game dynamics with a Gymnasium-style interaction contract."""

from math import isfinite
from random import Random

Cell = tuple[int, int]
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # north, east, south, west


def _move(cell: Cell, direction: int) -> Cell:
    dx, dy = DIRECTIONS[direction]
    return cell[0] + dx, cell[1] + dy


def potential_reward(reward: float, gamma: float, before: float,
                     after: float, *, terminated: bool) -> float:
    """Potential shaping, with zero continuation potential at true terminals."""
    if not all(isfinite(v) for v in (reward, gamma, before, after)):
        raise ValueError("shaping inputs must be finite")
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    return reward + gamma * (0.0 if terminated else after) - before


class BoardGame:
    """Shared episode lifecycle; time limits are external truncations."""

    def __init__(self, width: int, height: int, max_steps: int):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.width, self.height, self.max_steps = width, height, max_steps
        self.rng = Random()
        self.steps = 0
        self.done = True

    def _reset(self, seed: int | None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self.steps = 0
        self.done = False

    def _check_action(self, action: int) -> None:
        if self.done:
            raise RuntimeError("reset before stepping an ended episode")
        if type(action) is not int or not 0 <= action < self.action_dim:
            raise ValueError("invalid action")

    def _inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _finish(self, terminated: bool) -> tuple[bool, bool]:
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        self.done = terminated or truncated
        return terminated, truncated

    def close(self) -> None:
        pass


class Snake(BoardGame):
    """Relative actions: 0 left, 1 straight, 2 right; no reverse action."""

    action_dim = 3

    def __init__(self, size: int = 6, max_steps: int = 150):
        if type(size) is not int or not 4 <= size <= 32:
            raise ValueError("size must be an integer in [4, 32]")
        super().__init__(size, size, max_steps)
        self.observation_dim = 2 * size * size + 4
        self.body: list[Cell] = []
        self.heading = 1
        self.food: Cell | None = None
        self.score = 0

    def _spawn(self) -> Cell | None:
        occupied = set(self.body)
        free = [(x, y) for y in range(self.height) for x in range(self.width)
                if (x, y) not in occupied]
        return self.rng.choice(free) if free else None

    def reset(self, *, seed: int | None = None):
        self._reset(seed)
        y = self.height // 2
        self.body = [(2, y), (1, y), (0, y)]
        self.heading = 1
        self.score = 0
        self.food = self._spawn()
        return self.observation(), {"score": self.score}

    def observation(self) -> tuple[float, ...]:
        """Ordered body ranks, food plane, then one-hot heading."""
        n = self.width * self.height
        body = [0.0] * n
        food = [0.0] * n
        for rank, (x, y) in enumerate(self.body, start=1):
            body[y * self.width + x] = rank / n
        if self.food is not None:
            x, y = self.food
            food[y * self.width + x] = 1.0
        return tuple(body + food + [float(i == self.heading) for i in range(4)])

    def step(self, action: int):
        self._check_action(action)
        heading = (self.heading + action - 1) % 4
        head = _move(self.body[0], heading)
        eating = head == self.food
        # The tail vacates its cell on non-growth moves.
        blocked = self.body if eating else self.body[:-1]
        collision = not self._inside(head) or head in blocked
        if collision:
            terminated, truncated = self._finish(True)
            return self.observation(), -1.0, terminated, truncated, {
                "score": self.score, "reason": "collision",
            }
        self.heading = heading
        self.body.insert(0, head)
        if eating:
            self.score += 1
            self.food = self._spawn()
        else:
            self.body.pop()
        won = self.food is None
        terminated, truncated = self._finish(won)
        return self.observation(), 1.0 if eating else -0.01, terminated, truncated, {
            "score": self.score, "reason": "filled" if won else "",
        }

    def board(self) -> dict[Cell, str]:
        cells = dict.fromkeys(self.body, "body")
        if self.body:
            cells[self.body[0]] = "agent"
        if self.food is not None:
            cells[self.food] = "goal"
        return cells


DEFAULT_MAP = ("S....", ".##H.", "...#.", ".H...", "....G")


class HazardGrid(BoardGame):
    """Rectangular maps: S start, G goal, H hazard, # wall, . free."""

    action_dim = 4

    def __init__(self, layout: tuple[str, ...] = DEFAULT_MAP, max_steps: int = 80,
                 *, shaping_gamma: float | None = None):
        if (not layout or not layout[0] or any(len(r) != len(layout[0]) for r in layout)
                or any(c not in "SGH#." for r in layout for c in r)):
            raise ValueError("invalid rectangular grid layout")
        starts = [(x, y) for y, r in enumerate(layout) for x, c in enumerate(r)
                  if c == "S"]
        goals = [(x, y) for y, r in enumerate(layout) for x, c in enumerate(r)
                 if c == "G"]
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("layout must contain one start and one goal")
        super().__init__(len(layout[0]), len(layout), max_steps)
        if shaping_gamma is not None:
            potential_reward(0.0, shaping_gamma, 0.0, 0.0, terminated=False)
        self.layout = tuple(layout)
        self.start, self.goal = starts[0], goals[0]
        self.position = self.start
        self.shaping_gamma = shaping_gamma
        self.observation_dim = self.width * self.height

    def reset(self, *, seed: int | None = None):
        self._reset(seed)
        self.position = self.start
        return self.observation(), {}

    def observation(self) -> tuple[float, ...]:
        x, y = self.position
        return tuple(float(i == y * self.width + x)
                     for i in range(self.observation_dim))

    def potential(self, cell: Cell) -> float:
        return -float(abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1]))

    def step(self, action: int):
        self._check_action(action)
        before = self.position
        nxt = _move(before, action)
        if self._inside(nxt) and self.layout[nxt[1]][nxt[0]] != "#":
            self.position = nxt
        tile = self.layout[self.position[1]][self.position[0]]
        terminated, truncated = self._finish(tile in "GH")
        base = 1.0 if tile == "G" else -1.0 if tile == "H" else -0.04
        reward = base
        if self.shaping_gamma is not None:
            reward = potential_reward(base, self.shaping_gamma, self.potential(before),
                                      self.potential(self.position),
                                      terminated=terminated)
        return self.observation(), reward, terminated, truncated, {
            "base_reward": base, "success": tile == "G",
        }

    def board(self) -> dict[Cell, str]:
        colors = {"#": "wall", "G": "goal", "H": "hazard"}
        cells = {(x, y): colors[c] for y, row in enumerate(self.layout)
                 for x, c in enumerate(row) if c in colors}
        cells[self.position] = "agent"
        return cells


def make_game(name: str) -> Snake | HazardGrid:
    if name == "snake":
        return Snake()
    if name == "gridworld":
        return HazardGrid()
    raise ValueError(f"unknown game: {name}")
