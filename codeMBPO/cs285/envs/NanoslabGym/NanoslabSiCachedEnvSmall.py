import os
import sys
import pickle

sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project/Neural-Network-Materials')

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

class NanoslabSiCachedEnvSmall(gym.Env):
    def __init__(self, render_mode=None, n_range=(5, 11), a_range=(0.9, 1.035), rc=12.5, n_dos=200):
        self.rc = rc
        self.n_dos = n_dos

        self.n_range = n_range
        self.a_range = a_range

        self.d_n = 1
        self.d_a = 0.002

        self.n_n = int((n_range[1] - n_range[0]) / self.d_n) + 1
        self.n_a = int((a_range[1] - a_range[0]) / self.d_a)

        a = 5.431
        a_values = np.linspace(a_range[0], a_range[1], self.n_a) * a

        self.material: Type[Material] = SiliconNanoslab

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=0, high=self.n_a, shape=(n_dos + 2,), dtype=float)

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
        dos_directory = '/Users/krishnabhattaram/Desktop/CS285/Project/RunDataDOS/'
        self.cached_dos = {}
        self.cached_atoms_dicts = {}

        converter = AtomsConverter(environment_provider=AseEnvironmentProvider(5.0),
                                   device='cpu')

        print('Creating Cached DOS!')
        for n in tqdm(range(self.n_n)):
            for a in range(self.n_a):
                n_layers, a_val = n * self.d_n + self.n_range[0], a * self.d_a + self.a_range[0]
                filename = os.path.join(dos_directory,
                                        f"a={a_values[a]:.4f}_Ang_{n_layers}_layers_{1}x{0}x{0}.npz")
                data = np.load(filename)

                self.cached_dos[(n, a)] = data['dos']

                atom = Atoms(
                    symbols=[self.get_symbol_from_atomic_number(n) for n in data['atomic_numbers']],
                    positions=data['positions'],
                    cell=data['pv'],
                    pbc=[True, True, False])
                atom_dict = converter(atom)

                for k, v in atom_dict.items():
                    atom_dict[k] = v.squeeze(0)
                self.cached_atoms_dicts[(n, a)] = atom_dict

        self._agent_location = None
        self._target_location = None
        self._target_dos = None

    def get_atoms_dicts_from_obs(self, obs):
        return [self.cached_atoms_dicts[tuple(ob[:2])] for ob in obs]

    def get_symbol_from_atomic_number(self, atomic_number):
        number_to_symbol = {1: 'H', 2: 'He', 14: 'Si'}
        return number_to_symbol.get(atomic_number, -1)

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

    def reset(self, *, seed=None, return_info=False, options=None, ):
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
        return 1000 if np.array_equal(obs[:2], self._target_location) else 1 - np.linalg.norm(
            dos - self._target_dos) ** 2

    def _get_obs(self):
        return np.append(self._agent_location, self._target_dos)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "dos_norm_diff": np.linalg.norm(self._get_dos(self._agent_location) - self._target_dos) ** 2
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


# env = NanoslabCachedEnvSmall()
# obs0 = env.reset()
# print(env.get_atoms_from_obs([obs0]))
