import gym
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.walker2d import Walker2dEnv
from dm_control import suite
from dm_control.suite.wrappers import pixels


class GymWalkerDM(gym.Env):
    def __init__(self, frame_skip):
        self._env = suite.load(
            domain_name='walker', task_name='walk', task_kwargs={'random': 42})
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


class GymWalker(Walker2dEnv):
    def __init__(self):
        super().__init__()

class GymWalkerRNN(Walker2dEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_easy.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)
    
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward *= (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        done = not (height > 0.2 and height < 3.0 and
                    ang > -2.7 and ang < 2.7)
        ob = self._get_obs()
        return ob, reward, done, {}
    
    def reset_model(self):
        return super().reset_model()

# class GymWalkerDM(Walker2dEnv):
#     def __init__(self):
#         model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_dm.xml")
#         mujoco_env.MujocoEnv.__init__(self, model_path, 4)
#         utils.EzPickle.__init__(self)
# class GymWalkerDM1(Walker2dEnv):
#     def __init__(self):
#         model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_dm.xml")
#         mujoco_env.MujocoEnv.__init__(self, model_path, 1)
#         utils.EzPickle.__init__(self)
# class GymWalkerDM2(Walker2dEnv):
#     def __init__(self):
#         model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_dm.xml")
#         mujoco_env.MujocoEnv.__init__(self, model_path, 2)
#         utils.EzPickle.__init__(self)
# class GymWalkerDM3(Walker2dEnv):
#     def __init__(self):
#         model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_dm.xml")
#         mujoco_env.MujocoEnv.__init__(self, model_path, 3)
#         utils.EzPickle.__init__(self)

class GymWalkerDM1(GymWalkerDM):
    def __init__(self):
        super().__init__(1)
class GymWalkerDM2(GymWalkerDM):
    def __init__(self):
        super().__init__(2)
class GymWalkerDM3(GymWalkerDM):
    def __init__(self):
        super().__init__(3)

class GymWalkerDMVisual(Walker2dEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/walker2d_dm_visual.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)


class GymWalkerEasy(Walker2dEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_easy.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)


class GymWalkerBackwards(Walker2dEnv):
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_sourceulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = -1.0 * ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        return ob, reward, done, {}
