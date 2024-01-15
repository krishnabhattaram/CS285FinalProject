import os
import sys
import pickle

sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project/Neural-Network-Materials')

from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym

from cs285.agents.schnet_model import get_schnet_model
from cs285.infrastructure import pytorch_util as ptu

from schnetpack.environment import AseEnvironmentProvider
from schnetpack import AtomsConverter
from schnetpack import Properties


class DOSModelSchnet(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        num_dos: int,
        mpc_num_action_sequences: int,
        num_state: Optional[int] = 2,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1

        self.num_state = num_state
        self.num_dos = num_dos

        self.ensemble_size = ensemble_size
        self.dynamics_models = get_schnet_model(num_dos)
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "dos_mean", torch.zeros(self.num_dos, device=ptu.device)
        )
        self.register_buffer(
            "dos_std", torch.ones(self.num_dos, device=ptu.device)
        )
        self.register_buffer(
            "obs_mean", torch.zeros(self.num_state, device=ptu.device)
        )
        self.register_buffer(
            "obs_std", torch.ones(self.num_state, device=ptu.device)
        )

    def _collate_aseatoms(self, examples):
        """
        Build batch from systems and properties & apply padding

        Args:
            examples (list):

        Returns:
            dict[str->torch.Tensor]: mini-batch of atomistic systems
        """

        properties = examples[0]

        # initialize maximum sizes
        max_size = {
            prop: np.array(val.size(), dtype=int) for prop, val in properties.items()
        }

        # get maximum sizes
        for properties in examples[1:]:
            for prop, val in properties.items():
                max_size[prop] = np.maximum(
                    max_size[prop], np.array(val.size(), dtype=int)
                )

        # initialize batch
        batch = {
            p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
                examples[0][p].type()
            )
            for p, size in max_size.items()
        }
        has_atom_mask = Properties.atom_mask in batch.keys()
        has_neighbor_mask = Properties.neighbor_mask in batch.keys()

        if not has_neighbor_mask:
            batch[Properties.neighbor_mask] = torch.zeros_like(
                batch[Properties.neighbors]
            ).float()
        if not has_atom_mask:
            batch[Properties.atom_mask] = torch.zeros_like(batch[Properties.Z]).float()

        # If neighbor pairs are requested, construct mask placeholders
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Properties.neighbor_pairs_j in properties:
            batch[Properties.neighbor_pairs_mask] = torch.zeros_like(
                batch[Properties.neighbor_pairs_j]
            ).float()

        # build batch and pad
        for k, properties in enumerate(examples):
            for prop, val in properties.items():
                shape = val.size()
                s = (k,) + tuple([slice(0, d) for d in shape])
                batch[prop][s] = val

            # add mask
            if not has_neighbor_mask:
                nbh = properties[Properties.neighbors]
                shape = nbh.size()
                s = (k,) + tuple([slice(0, d) for d in shape])
                mask = nbh >= 0  # If not neighbor then index is -1 and masked
                batch[Properties.neighbor_mask][s] = mask
                batch[Properties.neighbors][s] = nbh * mask.long()

            if not has_atom_mask:
                z = properties[Properties.Z]
                shape = z.size()
                s = (k,) + tuple([slice(0, d) for d in shape])
                batch[Properties.atom_mask][s] = z > 0

            # Check if neighbor pair indices are present
            # Since the structure of both idx_j and idx_k is identical
            # (not the values), only one cutoff mask has to be generated
            if Properties.neighbor_pairs_j in properties:
                nbh_idx_j = properties[Properties.neighbor_pairs_j]
                shape = nbh_idx_j.size()
                s = (k,) + tuple([slice(0, d) for d in shape])
                batch[Properties.neighbor_pairs_mask][s] = nbh_idx_j >= 0

        return batch

    def update(self, i: int, obs: np.ndarray, dos: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            dos: (batch_size, dos_dim)
        """
        atom_dicts = self.env.get_atoms_dicts_from_obs(obs)
        dos = ptu.from_numpy(dos)

        norm_dos = (dos - self.dos_mean) / (self.dos_std + 1e-8)

        batch = self._collate_aseatoms(atom_dicts)

        predicted_norm_dos = self.dynamics_models(batch)['smooth_dos']
        loss = self.loss_fn(norm_dos, predicted_norm_dos)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, dos: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            dos: (n, num_dos)
        """
        obs = ptu.from_numpy(obs[:, :self.num_state])
        dos = ptu.from_numpy(dos)

        self.dos_std, self.dos_mean = torch.std_mean(dos, dim=0, keepdim=True)
        self.obs_std, self.obs_mean = torch.std_mean(obs, dim=0, keepdim=True)

    @torch.no_grad()
    def get_dos_predictions(
        self, i: int, obs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_model

        Args:
            i:   Which dynamics model to choose
            obs: (batch_size, ob_dim)
        Returns: (batch_size, num_dos)
        """
        atom_dicts = self.env.get_atoms_dicts_from_obs(obs)
        batch = self._collate_aseatoms(atom_dicts)

        predicted_norm_dos = self.dynamics_models(batch)['smooth_dos']
        predicted_dos = (predicted_norm_dos * (self.dos_std + 1e-8)) + self.dos_mean

        return ptu.to_numpy(predicted_dos)

    @torch.no_grad()
    def get_dos_ensemble_prediction(self, obs: np.ndarray):
        return self.get_dos_predictions(0, obs).squeeze()
