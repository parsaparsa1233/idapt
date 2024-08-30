import os
import gym

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import pixels


class GymHalfCheetahDM_(gym.Env):
    def __init__(self, frame_skip):
        self._env = suite.load(
            domain_name='cheetah', task_name='run', task_kwargs={'random': 42})
        # self._env = pixels.Wrapper(self._env)
        self.frame_skip = frame_skip
        self.max_episode_length = 1000
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_size,), dtype=np.float32)


    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        return np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0)

    def step(self, action):
        reward = 0
        for k in range(self.frame_skip):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1  # Increment internal timer
            done = state.last()
            if done:
                self.reset()
                break
        observation = np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0)
        return observation, reward, done, {}

    def render(self, mode="rgb_array", height=64, width=64, camera_id=0):
        return self._env.physics.render(height=height, width=width, camera_id=0)

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()])

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]


class GymHalfCheetahDM(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/half_cheetah_dm.xml"
        )
        super().__init__(xml_file=model_path)

class GymHalfCheetahDM1(GymHalfCheetahDM_):
    def __init__(self):
        super().__init__(1)
class GymHalfCheetahDM2(GymHalfCheetahDM_):
    def __init__(self):
        super().__init__(2)
class GymHalfCheetahDM3(GymHalfCheetahDM_):
    def __init__(self):
        super().__init__(3)
class GymHalfCheetahDM4(GymHalfCheetahDM_):
    def __init__(self):
        super().__init__(4)

class GymHalfCheetahDMVisual(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/half_cheetah_dm_visual.xml"
        )
        super().__init__(xml_file=model_path)


class GymHalfCheetahEasy(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/half_cheetah_easy.xml"
        )
        super().__init__(xml_file=model_path)
