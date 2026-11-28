"""Optional Gymnasium adapter and Pygame display for the custom games."""

import gymnasium as gym
import numpy as np

from prometheus_rl.environments.games import make_game

COLORS = {
    "wall": (72, 79, 92), "goal": (237, 172, 64), "hazard": (206, 65, 77),
    "body": (73, 160, 105), "agent": (53, 109, 198),
}


class GameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, name: str, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("unsupported render mode")
        self.game = make_game(name)
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(self.game.action_dim)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.game.observation_dim,), dtype=np.float32
        )
        self._screen = None
        self._clock = None
        self._closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.game.reset(seed=seed)
        if self.render_mode == "human":
            self.render()
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        obs, reward, terminated, truncated, info = self.game.step(int(action))
        if self.render_mode == "human":
            self.render()
        return np.asarray(obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        scale = 48
        frame = np.full((self.game.height * scale, self.game.width * scale, 3),
                        235, dtype=np.uint8)
        for (x, y), kind in self.game.board().items():
            frame[y * scale + 2:(y + 1) * scale - 2,
                  x * scale + 2:(x + 1) * scale - 2] = COLORS[kind]
        frame[::scale, :, :] = 205
        frame[:, ::scale, :] = 205
        if self.render_mode == "rgb_array":
            return frame
        if self._closed:
            return None
        import pygame

        if self._screen is None:
            pygame.display.init()
            self._screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            pygame.display.set_caption("Prometheus-RL | Custom game")
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
        self._screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.display.quit()
            self._screen = None
        self._closed = True
        self.game.close()
