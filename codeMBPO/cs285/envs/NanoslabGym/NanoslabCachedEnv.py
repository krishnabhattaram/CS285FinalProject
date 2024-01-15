import os
import sys
import pickle

sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project/Neural-Network-Materials')

import os
from typing import Type

import gym
import scipy
from gym import spaces
import numpy as np
from gym.spaces import Tuple

from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.SiliconNanoslab import SiliconNanoslab
from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.helpers import get_dos
from codeMBPO.cs285.envs.NanoslabGym.Nanoslabs.material import Material

from tqdm import tqdm
from ase import Atoms
from schnetpack.environment import AseEnvironmentProvider
from schnetpack import AtomsConverter
from schnetpack import Properties

class NanoslabCachedEnv(gym.Env):
    def __init__(self, render_mode=None, n_range=(3, 20), a_range=(3, 9), rc=12.5, n_dos=10):
        self.rc = rc
        self.n_dos = n_dos

        self.n_range = n_range
        self.a_range = a_range

        self.d_n = 1
        self.d_a = 0.1

        self.n_n = int((n_range[1] - n_range[0]) / self.d_n) + 1
        self.n_a = int((a_range[1] - a_range[0]) / self.d_a) + 1

        self.material: Type[Material] = SiliconNanoslab

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=0, high=self.n_a, shape=(n_dos+2,), dtype=float)
        #spaces.Dict({
        #     "agent":  spaces.Box(low=np.array([0, 0]), high=np.array([self.n_a-1, self.n_n-1]),
        #                          shape=(2,), dtype=np.float32),
        #     "target":  spaces.Box(low=0, high=1.0, shape=(n_dos,), dtype=np.float32)})

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Loading DOS from database
        dos_directory = '/Users/krishnabhattaram/Desktop/CS285/Project/SiliconDatabase/'
        self.cached_dos = {}
        self.cached_atoms_dicts = {}

        converter = AtomsConverter(environment_provider=AseEnvironmentProvider(5.0),
                                   device='cpu')

        print('Creating Cached DOS!')
        for n in tqdm(range(self.n_n)):
            for a in range(self.n_a):
                n_layers, a_val = n * self.d_n + self.n_range[0], a * self.d_a + self.a_range[0]
                filename = os.path.join(dos_directory, f'Si_a={round(a_val, 4)}_n={n_layers}.npz')

                data = np.load(filename)
                assert data['n_layers'] == n_layers
                assert np.abs(data['a_val'] - a_val) < 1e-10

                self.cached_dos[(n, a)] = get_dos(data['energies'], n_bins=self.n_dos)

                atom = Atoms(
                    symbols=[14 for _ in range(n_layers)],
                    positions=[(0, 0, z*a_val) for z in range(n_layers)],
                    cell=[[a_val, 0, 0], [0, a_val, 0], [0, 0, n_layers*a_val]],
                    pbc=[True, True, False])
                atom_dict = converter(atom)

                for k, v in atom_dict.items():
                    atom_dict[k] = v.squeeze(0)
                self.cached_atoms_dicts[(n, a)] = atom_dict

        self._agent_location = None
        self._target_location = None
        self._target_dos = None

    def get_atoms_from_obs(self, obs):
        return [self.cached_atoms_dicts[tuple(ob[:2])] for ob in obs]

    def _get_dos(self, location):
        dos = self.cached_dos.get(tuple(location), None)
        assert dos is not None
        return dos

    def get_dos(self, obs):
        """
        Get DOS for a batch of observations
        obs: n_batch x (2 + n_dos)
        """
        return np.array([self._get_dos(ob[:2]) for ob in obs])

    def reset(self, *, seed=None, return_info=False, options=None,):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, return_info=return_info, options=options)

        # Choose the agent's location uniformly at random
        space = spaces.Box(low=np.array([0, 0]), high=np.array([self.n_n - 1, self.n_a - 1]), dtype=int, seed=seed)

        self._target_location = space.sample()
        self._target_dos = self._get_dos(self._target_location)

        self._agent_location = space.sample()
        while np.array_equal(self._target_location, self._agent_location):
            self._agent_location = space.sample()

        observation = self._get_obs()

        return observation

    def get_reward(self, obs, dos):
        return 1000 if np.array_equal(obs[:2], self._target_location) else 1 - np.linalg.norm(dos - self._target_dos)**2

    def _get_obs(self):
        return np.append(self._agent_location, self._target_dos)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "dos_norm_diff": np.linalg.norm(self._get_dos(self._agent_location) - self._target_dos)**2
        }

    def get_next_observation(self, ob, ac):
        direction = self._action_to_direction[ac]

        # We use `np.clip` to make sure we don't leave the grid
        next_loc = np.clip(
            ob[:2] + direction, np.array([0, 0]), np.array([self.n_n - 1, self.n_a - 1])
        )

        return np.append(next_loc, self._target_dos)

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, np.array([0, 0]), np.array([self.n_n - 1, self.n_a - 1])
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()
        reward = self.get_reward(observation, self._get_dos(self._agent_location))

        info = self._get_info()

        return observation, reward, terminated, info
