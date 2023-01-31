import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque

class SnakeWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "all" : spaces.Box(0, 3, (size * size + 4, ))
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Box(0, 3, (2, ), int)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def step(self, action):
        self.step_count += 1

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action1, action2 = action

        if not action1 % 2 == self.snake1_action % 2:
            self.snake1_action = action1

        if not action2 % 2 == self.snake2_action % 2:
            self.snake2_action = action2

        direction1 = self._action_to_direction[self.snake1_action]
        direction2 = self._action_to_direction[self.snake2_action]

        head1 = np.array(self.snake1[len(self.snake1) - 1] + direction1)
        self.snake1.append(head1)
        self.head1 = head1

        head2 = np.array(self.snake2[len(self.snake2) - 1] + direction2)
        self.snake2.append(head2)
        self.head2 = head2

        #check if a snake hit an apple
        redo_apple = False
        reward1_for_apple = 0
        reward2_for_apple = 0
        if all(np.equal(head1, self.apple)):
            redo_apple = True
            self.snake1_length += 1
            reward1_for_apple = 5

        if all(np.equal(head2, self.apple)):
            redo_apple = True
            self.snake2_length += 1
            reward2_for_apple = 5

        #make the snakes the right size
        if len(self.snake1) > self.snake1_length:
            self.snake1.popleft()

        if len(self.snake2) > self.snake2_length:
            self.snake2.popleft()
        
        if redo_apple:
            self._generate_apple()

        self.snake1.pop()
        self.snake2.pop()
        died1 = any(head1 < 0) or any(head1 >= self.size) or any([all(np.equal(head1, arr)) for arr in self.snake1]) or any([all(np.equal(head1, arr)) for arr in self.snake2])
        died2 = any(head2 < 0) or any(head2 >= self.size) or any([all(np.equal(head2, arr)) for arr in self.snake1]) or any([all(np.equal(head2, arr)) for arr in self.snake2])
        self.snake1.append(head1)
        self.snake2.append(head2)
        self.died1 = died1
        self.died2 = died2

        if all(np.equal(head1, head2)):
            self.died1 = True
            self.died2 = True

        terminated = 0
        #Manhattan distance, -0.2 pnealty, so they won't stay in one place
        distance1_from_apple = np.sum(np.abs(self.apple - head1))
        distance2_from_apple = np.sum(np.abs(self.apple - head2))
        reward = ( (1 / distance1_from_apple) + reward1_for_apple, (1 / distance2_from_apple) + reward2_for_apple)

        observation = self._get_obs()
        info = self._get_info()
        if (died1 and died2):
            terminated = 1
            reward = (-500, -500)  
        elif died1:
            terminated = 1
            reward = (-500, 500)
        elif died2:
            terminated = 1
            reward = (500, -500)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        map = np.zeros((self.size, self.size))
        if self.died1 or self.died2:
            return {"all": np.append(np.append(map.flatten(), self.head1), self.head2)}

        for c in self.snake1:
            map[c] = 1
        for c in self.snake2:
            map[c] = 2
        map[self.apple] = 3
        return {"all": np.append(np.append(map.flatten(), self.head1), self.head2)}

    def _get_info(self):
        return {}

    def _generate_apple(self):
        self.apple = np.random.random_integers(0, self.size - 1, (2,))

        while any(all(np.equal(self.apple, arr)) for arr in self.snake1) or any(all(np.equal(self.apple, arr)) for arr in self.snake2):
            self.apple = np.random.random_integers(0, self.size - 1, (2,))

        

    def reset(self):
        super().reset()

        self.step_count = 0

        self.snake1 = deque([np.array([0, 0]), np.array([0, 1]), np.array([0, 2])], 100)
        n = self.size
        self.snake2 = deque([np.array([n-1, n-1]), np.array([n-1, n-2]), np.array([n-1, n-3])], 100)

        self.snake1_action = 1
        self.snake2_action = 3

        self.snake1_length = 3
        self.snake2_length = 3
        self.head1 = self.snake1[len(self.snake1) - 1]
        self.head2 = self.snake1[len(self.snake1) - 1]

        
        self.died1 = False
        self.died2 = False

        self._generate_apple()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        #draw apple
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.apple + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        #draw first snake
        for c in self.snake1:
            pygame.draw.rect(
                canvas,
                (125, 125, 0),
                pygame.Rect(
                    pix_square_size * c,
                    (pix_square_size, pix_square_size),
                ),
            )

        for c in self.snake2:
            pygame.draw.rect(
                canvas,
                (0, 250, 0),
                pygame.Rect(
                    pix_square_size * c,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

