from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class DOSModel(nn.Module):
    def __init__(
            self,
            env: gym.Env,
            make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
            make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
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
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.num_state,
                    self.num_dos,
                )
                for _ in range(ensemble_size)
            ]
        )
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

    def update(self, i: int, obs: np.ndarray, dos: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            dos: (batch_size, dos_dim)
        """
        obs = ptu.from_numpy(obs[:, :self.num_state])
        dos = ptu.from_numpy(dos)

        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) 
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        norm_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        norm_dos = (dos - self.dos_mean) / (self.dos_std + 1e-8)

        predicted_norm_dos = self.dynamics_models[i](norm_obs)

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
        # TODO(student): update the statistics
        self.dos_std, self.dos_mean = torch.std_mean(dos, dim=0, keepdim=True)
        self.obs_std, self.obs_mean = torch.std_mean(obs, dim=0, keepdim=True)

    @torch.no_grad()
    def get_dos_predictions(
            self, i: int, obs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            i:   Which dynamics model to choose
            obs: (batch_size, ob_dim)
        Returns: (batch_size, num_dos)
        """
        obs = ptu.from_numpy(obs[:, :self.num_state])
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        norm_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)

        predicted_norm_dos = self.dynamics_models[i](norm_obs)
        predicted_dos = (predicted_norm_dos * (self.dos_std + 1e-8)) + self.dos_mean

        return ptu.to_numpy(predicted_dos)

    @torch.no_grad()
    def get_dos_ensemble_prediction(self, obs: np.ndarray):
        return np.mean([self.get_dos_predictions(i, obs) for
                        i in range(self.ensemble_size)], axis=0).squeeze()