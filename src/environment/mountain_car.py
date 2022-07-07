from dana_codes.src.base import Environment
import gym
import numpy as np
import math
from typing import Optional

from gym import spaces
from gym.error import DependencyNotInstalled


class MountainCarEnv(Environment):
    def __init__(self, name='MountainCar-v0', continuous=True, granularity=50, max_steps=200):
        self.discrete_obs_step = None
        self.discrete_os_size = None
        self.env = None
        self.granularity = None
        self.max_episode_steps = None
        self.continuous = continuous
        self.set_environment(name)
        self.set_granularity(granularity)
        self.set_max_steps(max_steps)

    def get_action_shape(self):
        return self.env.action_space.shape

    def get_obs_shape(self):
        return self.env.observation_space.shape

    def set_environment(self, en):
        self.env = gym.make(en)

    def set_granularity(self, gr):
        self.granularity = gr
        self.discrete_os_size = [gr] * len(self.env.observation_space.high)
        self.discrete_obs_step = (self.env.observation_space.high - self.env.observation_space.low) \
                                 / self.discrete_os_size

    def set_max_steps(self, ms=-1):
        self.env._max_episode_steps = ms
        self.max_episode_steps = ms

    def discretize(self, obs):
        d_obs = (obs - self.env.observation_space.low) / self.discrete_obs_step
        return d_obs.astype(int).astype(float)

    def get_n_actions(self):
        return self.env.action_space.n

    def reset(self):
        if self.continuous:
            return self.env.reset()
        else:
            return self.discretize(self.env.reset())

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if self.continuous:
            n_s = next_state
        else:
            n_s = self.discretize(next_state)

        return n_s, reward, done, info

    def get_goal_position(self):
        if self.continuous:
            return self.env.goal_position
        else:
            return self.discretize(np.array([self.env.goal_position, self.env.observation_space.low[1]]))[0]

    def is_goal(self, state):
        return state[0] >= self.get_goal_position()

    def render(self):
        self.env.render()


"""
    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                                                 | Min                | Max    | Unit |
    |-----|-------------------------------------------------------------|--------------------|--------|------|
    | 0   | position of the car along the x-axis                        | -Inf               | Inf    | position (m) |
    | 1   | velocity of the car                                         | -Inf               | Inf  | position (m) |

    ### Action Space

    There are 3 discrete deterministic actions:

    | Num | Observation                                                 | Value   | Unit |
    |-----|-------------------------------------------------------------|---------|------|
    | 0   | Accelerate to the left                                      | Inf    | position (m) |
    | 1   | Don't accelerate                                            | Inf  | position (m) |
    | 2   | Accelerate to the right                                     | Inf    | position (m) |

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and velocity is clipped to the range `[-0.07, 0.07]`.


    ### Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is penalised with a reward of -1 for each timestep it isn't at the goal and is not penalised (reward = 0) for when it reaches the goal.

    ### Starting State

    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*. The starting velocity of the car is always assigned to 0.

    ### Episode Termination

    The episode terminates if either of the following happens:
    1. The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. The length of the episode is 200.

 """


class MountainCar3DEnv(Environment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        self.max_steps = 400
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, self.min_position, -self.max_speed, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_position, self.max_speed, self.max_speed], dtype=np.float32)

        self.state = None
        self.info = None
        self.screen = None
        self.scr_width = 1200
        self.scr_height = 400
        self.car_width = 40
        self.car_height = 20
        self.clock = None
        self.is_open = True
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.info_space = (6,)

        self.step_counter = 0

    def update_position(self):
        x_pos, y_pos, x_vel, y_vel = self.state
        x_pos += x_vel
        y_pos += y_vel

        x_pos = np.clip(x_pos, self.min_position, self.max_position)
        y_pos = np.clip(y_pos, self.min_position, self.max_position)

        if x_pos == self.max_position and x_vel > 0:
            x_vel = 0
        if y_pos == self.min_position and y_vel > 0:
            y_vel = 0
        if x_pos == self.min_position and x_vel < 0:
            x_vel = 0
        if y_pos == self.min_position and y_vel < 0:
            y_vel = 0

        self.state = x_pos, y_pos, x_vel, y_vel

    def update_velocity(self, action):
        x_pos, y_pos, x_vel, y_vel = self.state

        x_update, y_update = 0, 0
        if action == 0:
            x_update, y_update = 0, 0
        elif action == 1:
            x_update, y_update = -1, 0
        elif action == 2:
            x_update, y_update = +1, 0
        elif action == 3:
            x_update, y_update = 0, -1
        elif action == 4:
            x_update, y_update = 0, +1

        x_vel += np.clip(x_update * self.force + math.cos(3 * x_pos) * (-self.gravity), -self.max_speed, self.max_speed)
        y_vel += np.clip(y_update * self.force + math.cos(3 * y_pos) * (-self.gravity), -self.max_speed, self.max_speed)

        self.state = x_pos, y_pos, x_vel, y_vel

    def check_termination(self):
        x_pos, y_pos, x_vel, y_vel = self.state

        if x_pos > self.goal_position and y_pos > self.goal_position:
            return True
        elif self.step_counter >= self.max_steps:
            return True

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        self.step_counter += 1

        self.update_velocity(action)
        self.update_position()

        done = self.check_termination()

        reward = -1.0

        # info = (position, velocity, self.height(position), self.goal_position, self.height(self.goal_position),
        # reward, self.diver_height(position), self.force, self.gravity)

        # self.info = (info[0], info[1], info[2], info[5], info[6], info[7])

        return np.array(self.state, dtype=np.float32), reward, done, None

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            fix_start_point: bool = False,
            options: Optional[list] = None,
    ):

        if fix_start_point:
            self.state = np.array(options)
        else:
            self.state = np.array([np.random.uniform(low=-0.6, high=-0.4),
                                   np.random.uniform(low=-0.6, high=-0.4), 0, 0])

        position = self.state[0]
        velocity = 0

        self.step_counter = 0

        info = (position, velocity, self.height(position), self.goal_position, self.height(self.goal_position),
                0, self.diver_height(position), self.force, self.gravity)

        self.info = (info[0], info[1], info[2], info[5], info[6], info[7])

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), np.array(self.info, dtype=np.float32)

    @staticmethod
    def _height_x(xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    @staticmethod
    def _height_y(ys):
        return np.sin(3 * ys) * 0.45 + 0.55

    @staticmethod
    def height(xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    @staticmethod
    def diver_height(xs):
        return 3 * np.cos(3 * xs) * 0.45

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.scr_width, self.scr_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        x_pos, y_pos, x_vel, y_vel = self.state
        x_surf = self._render_dim(x_pos, MountainCar3DEnv._height_x)
        y_surf = self._render_dim(y_pos, MountainCar3DEnv._height_y)

        self.screen.blit(x_surf, (0, 0))
        self.screen.blit(y_surf, (self.scr_width / 2, 0))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.is_open

    # def get_keys_to_action(self):
    #     # Control with left and right arrow keys.
    #     return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def _render_dim(self, pos, height_fn):
        import pygame
        from pygame import gfxdraw

        world_width = self.max_position - self.min_position
        scale = self.scr_width / (2 * world_width)

        surf = pygame.Surface((self.scr_width / 2, self.scr_height))
        surf.fill((255, 255, 255))

        # Draw the track curve
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = height_fn(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(surf, points=xys, closed=False, color=(0, 0, 0))

        # Draw the car
        clearance = 10
        l, r, t, b = -self.car_width / 2, self.car_width / 2, self.car_height, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + height_fn(pos) * scale,
                )
            )

        gfxdraw.aapolygon(surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, coords, (0, 0, 0))

        # Draw the wheels
        for c in [(self.car_width / 4, 0), (-self.car_width / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + height_fn(pos) * scale),
            )

            gfxdraw.aacircle(
                surf, wheel[0], wheel[1], int(self.car_height / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                surf, wheel[0], wheel[1], int(self.car_height / 2.5), (128, 128, 128)
            )

        # Draw the flag
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(height_fn(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        # Make it human friendly
        surf = pygame.transform.flip(surf, False, True)

        return surf

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.is_open = False

    def get_n_actions(self):
        return 5

    def get_action_shape(self):
        return ()

    def get_obs_shape(self):
        return 4,
