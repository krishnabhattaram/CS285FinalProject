from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from cs285.env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
import cs285.infrastructure.pytorch_util as ptu

def dqn_nanoslab_config(
    num_agent_train_steps_per_iter: int = 1000,
    exp_name: Optional[str] = None,
    replay_buffer_capacity: int = 1000000,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 3000000,
    discount: float = 0.99,
    target_update_period: int = 1000,
    clip_grad_norm: Optional[float] = None,
    use_double_q: bool = False,
    mbpo_rollout_length: int = 0,
    batch_size: int = 128,
    **kwargs
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        return ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=num_actions,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (total_steps * 0.1, 0.02),
        ],
        outside_value=0.02,
    )

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "num_agent_train_steps_per_iter": num_agent_train_steps_per_iter,
        "batch_size": batch_size,
        "replay_buffer_capacity": replay_buffer_capacity,
        "mbpo_rollout_length": mbpo_rollout_length,
    }