from typing import Type

import gym
import scipy
from gym import spaces
import numpy as np

from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.SiliconNanoslab import SiliconNanoslab
from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.helpers import get_dos
from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.material import Material

from tqdm import tqdm

class BulkEnv(gym.Env):
    def __init__(self, render_mode=None, a_range=(1, 12), rc=12.5):
        self.a_range = a_range
        self.rc = rc

        self.d_a = 0.1
        self.n_a = int((a_range[1] - a_range[0])/self.d_a) + 1

        self.material: Type[Material] = SiliconNanoslab

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=0, high=self.n_a, dtype=int)

        # We have 2 actions, corresponding to "left", "right"
        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([-1]),
            1: np.array([1]),
        }


    def _get_energies(self, location):
        n_layers, a_val = location[0] * self.d_n + self.n_range[0], location[1] * self.d_a + self.a_range[0]
        instance = self.material(n_layers=n_layers, a=a_val, rc=self.rc)

        return instance.get2DPeriodicEnergies_k(N=10).flatten()

    def _get_KS_difference_location(self, location):
        energies_agent = self._get_energies(location)
        energies_target = self._target_energies

        return 10 * (0.5 - scipy.stats.ks_2samp(energies_agent, energies_target).statistic)

    def _get_KS_difference(self):
        cached_value = self.cached_rewards.get(tuple(self._agent_location), None)
        if cached_value is not None:
            return cached_value

        return self._get_KS_difference_location(self._agent_location)

    def _get_dos(self, location):
        return get_dos(self._get_energies(location))

    def _get_obs(self):
        return np.append(self._agent_location, self._target_dos)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "KS_difference": self._get_KS_difference()
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        space = spaces.Box(low=0, high=self.n_a, dtype=int, seed=seed)
        self._agent_location = space.sample()

        self._target_location = self.observation_space.sample()
        self._target_energies = self._get_energies(self._target_location)
        self._target_dos = self._get_dos(self._target_location)
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = space.sample()

        observation = self._get_obs()

        return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, np.array([0, 0]), np.array([self.n_n - 1, self.n_a - 1])
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 100 if terminated else self._get_KS_difference()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, info
