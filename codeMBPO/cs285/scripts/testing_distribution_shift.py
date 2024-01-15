import sys

sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project/codeMBPO')
sys.path.append('/Users/krishnabhattaram/Desktop/CS285/Project')

import os
import time
from typing import Optional
from matplotlib import pyplot as plt
import yaml
from cs285 import envs

from cs285.agents.dos_model import DOSModel
from cs285.agents.dos_model_schnet import DOSModelSchnet

from codeDQN.cs285.agents.dqn_agent import DQNAgent
from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from scripting_utils import make_logger, make_config

import argparse

from cs285.envs import register_envs

register_envs()


def run_training_loop(
        config: dict, logger: Logger, args: argparse.Namespace, dqn_config: Optional[dict]
):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    small_env = gym.make('NanoslabSiCachedEnvSmall-v0', render_mode=None)
    large_env = gym.make('NanoslabSiCachedEnvLarge-v0', render_mode=None)

    dense_model = DOSModel(
                        small_env,
                        num_dos=200,
                        **config["agent_kwargs"],
                    )

    # initialize agent
    schnet_dos_model = DOSModelSchnet(
        small_env,
        num_dos=200,
        **config["agent_kwargs"],
    )

    print(f'dense Model has {sum(p.numel() for p in dense_model.parameters() if p.requires_grad)} parameters')
    print(f'schnet Model has {sum(p.numel() for p in schnet_dos_model.parameters() if p.requires_grad)} parameters')

    small_replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])
    large_replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    ep_len = config["ep_len"] or small_env.spec.max_episode_steps

    print("Collecting data...")
    # TODO(student): collect at least config["initial_batch_size"] transitions with a random policy
    # HINT: Use `utils.RandomPolicy` and `utils.sample_trajectories`
    trajs, envsteps_this_batch = utils.sample_trajectories(env=small_env,
                                                           policy=utils.RandomPolicy(env=small_env),
                                                           min_timesteps_per_batch=config["initial_batch_size"],
                                                           max_length=ep_len,
                                                           render=False)

    # insert newly collected data into replay buffer
    for traj in trajs:
        small_replay_buffer.batched_insert(
            observations=traj["observation"],
            actions=traj["action"],
            rewards=traj["reward"],
            next_observations=traj["next_observation"],
            dones=traj["done"],
        )


    # update agent's statistics with the entire replay buffer
    dense_model.update_statistics(
        obs=small_replay_buffer.observations[: len(small_replay_buffer)],
        dos=small_env.get_dos(small_replay_buffer.observations[: len(small_replay_buffer)]),
    )

    # update agent's statistics with the entire replay buffer
    schnet_dos_model.update_statistics(
        obs=small_replay_buffer.observations[: len(small_replay_buffer)],
        dos=small_env.get_dos(small_replay_buffer.observations[: len(small_replay_buffer)]),
    )

    # train agent
    print("Training agent...")
    all_losses_dense = []
    #all_losses_sch = []
    for _ in tqdm.trange(
            config["num_agent_train_steps_per_iter"], dynamic_ncols=True
    ):
        step_losses_dense = []
        #step_losses_sch = []

        #transitions = small_replay_buffer.sample(config["train_batch_size"])
        #step_losses_sch += [schnet_dos_model.update(i=0, obs=transitions["observations"],
        #                                     dos=small_env.get_dos(transitions["observations"]))]

        for i in range(dense_model.ensemble_size):
            transitions = small_replay_buffer.sample(config["train_batch_size"])
            step_losses_dense += [dense_model.update(i=i, obs=transitions["observations"],
                                             dos=small_env.get_dos(transitions["observations"]))]

        #all_losses_sch.append(np.mean(step_losses_sch))
        all_losses_dense.append(np.mean(step_losses_dense))

    # on iteration 0, plot the full learning curve
    #plt.plot(all_losses_sch, label='SchNet Small Environment Loss')
    # plt.plot(all_losses_dense, label='Dense Layer Small Environment Loss')
    # plt.title(f"Dynamics Model Training Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Step")
    # plt.legend()
    # plt.show()

    loss_small = all_losses_dense

    layer_err = []
    layer = []

    for n_layers in range(5, 12):
        n = int((n_layers - 5) / 1)
        a_val = 1.0
        a = int((a_val - 0.9)/0.002)

        obs = np.append([n, a], 200*[1])[None, :]
        dos_obs = small_env.get_dos(obs).squeeze()
        dos_pred = dense_model.get_dos_ensemble_prediction(obs).squeeze()

        layer += [n_layers]
        layer_err += [np.linalg.norm(dos_pred - dos_obs)]

        # plt.figure()
        # plt.plot(dos_obs, label='Actual')
        # plt.plot(dos_pred, label='Predicted')
        # plt.title(f'{n_layers} layers')
        # plt.legend()
        # plt.show()

    for n_layers in range(13, 20):
        n = int((n_layers - 13) / 1)
        a_val = 1.0
        a = int((a_val - 0.9)/0.002)

        obs = np.append([n, a], 200*[1])[None, :]
        dos_obs = large_env.get_dos(obs).squeeze()
        dos_pred = dense_model.get_dos_ensemble_prediction(obs).squeeze()

        layer += [n_layers]
        layer_err += [np.linalg.norm(dos_pred - dos_obs)]

        # plt.figure()
        # plt.title(f'{n_layers} layers')
        # plt.plot(dos_obs, label='Actual')
        # plt.plot(dos_pred, label='Predicted')
        # plt.legend()
        # plt.show()

    # plt.figure()
    # plt.title('DOS Error for Unstrained Structure by Layer (before training on large structures)')
    # plt.xlabel('Number of Layers')
    # plt.ylabel('Prediction Error')
    # plt.plot(layer, layer_err)
    # plt.scatter(layer, layer_err)
    # plt.show()

    layer_num_small = layer
    layer_err_small = layer_err




    ### LARGE ENV ###

    print("Collecting data...")
    # TODO(student): collect at least config["initial_batch_size"] transitions with a random policy
    # HINT: Use `utils.RandomPolicy` and `utils.sample_trajectories`
    trajs, envsteps_this_batch = utils.sample_trajectories(env=large_env,
                                                           policy=utils.RandomPolicy(env=small_env),
                                                           min_timesteps_per_batch=config["initial_batch_size"],
                                                           max_length=ep_len,
                                                           render=False)

    # insert newly collected data into replay buffer
    for traj in trajs:
        large_replay_buffer.batched_insert(
            observations=traj["observation"],
            actions=traj["action"],
            rewards=traj["reward"],
            next_observations=traj["next_observation"],
            dones=traj["done"],
        )

    # update agent's statistics with the entire replay buffer
    dense_model.update_statistics(
        obs=small_replay_buffer.observations[: len(small_replay_buffer)],
        dos=small_env.get_dos(small_replay_buffer.observations[: len(small_replay_buffer)]),
    )

    schnet_dos_model.update_statistics(
        obs=small_replay_buffer.observations[: len(small_replay_buffer)],
        dos=small_env.get_dos(small_replay_buffer.observations[: len(small_replay_buffer)]),
    )

    # train agent
    print("Training agent...")
    all_losses_dense = []
    # all_losses_sch = []
    for _ in tqdm.trange(
            config["num_agent_train_steps_per_iter"], dynamic_ncols=True
    ):
        step_losses_dense = []
        # step_losses_sch = []

        # transitions = large_replay_buffer.sample(config["train_batch_size"])
        # step_losses_sch += [schnet_dos_model.update(i=0, obs=transitions["observations"],
        #                                     dos=large_env.get_dos(transitions["observations"]))]

        for i in range(dense_model.ensemble_size):
            transitions = large_replay_buffer.sample(config["train_batch_size"])
            step_losses_dense += [dense_model.update(i=i, obs=transitions["observations"],
                                                     dos=large_env.get_dos(transitions["observations"]))]

        # all_losses_sch.append(np.mean(step_losses_sch))
        all_losses_dense.append(np.mean(step_losses_dense))

    # plt.plot(all_losses_sch, label='SchNet Small Environment Loss')
    # plt.plot(all_losses_dense, label='Dense Layer Large Environment Loss')
    # plt.title(f"Dynamics Model Training Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Step")
    # plt.legend()
    # plt.show()

    loss_large = all_losses_dense

    layer_err = []
    layer = []

    for n_layers in range(5, 12):
        n = int((n_layers - 5) / 1)
        a_val = 1.0
        a = int((a_val - 0.9) / 0.002)

        obs = np.append([n, a], 200 * [1])[None, :]
        dos_obs = small_env.get_dos(obs).squeeze()
        dos_pred = dense_model.get_dos_ensemble_prediction(obs).squeeze()

        layer += [n_layers]
        layer_err += [np.linalg.norm(dos_pred - dos_obs)]

        # plt.figure()
        # plt.plot(dos_obs, label='Actual')
        # plt.plot(dos_pred, label='Predicted')
        # plt.title(f'{n_layers} layers')
        # plt.legend()
        # plt.show()

    for n_layers in range(13, 20):
        n = int((n_layers - 13) / 1)
        a_val = 1.0
        a = int((a_val - 0.9) / 0.002)

        obs = np.append([n, a], 200 * [1])[None, :]
        dos_obs = large_env.get_dos(obs).squeeze()
        dos_pred = dense_model.get_dos_ensemble_prediction(obs).squeeze()

        layer += [n_layers]
        layer_err += [np.linalg.norm(dos_pred - dos_obs)]

        # plt.figure()
        # plt.title(f'{n_layers} layers')
        # plt.plot(dos_obs, label='Actual')
        # plt.plot(dos_pred, label='Predicted')
        # plt.legend()
        # plt.show()

    # plt.figure()
    # plt.title('DOS Error for Unstrained Structure by Layer (after training on large structures)')
    # plt.xlabel('Number of Layers')
    # plt.ylabel('Prediction Error')
    # plt.plot(layer, layer_err)
    # plt.scatter(layer, layer_err)
    # plt.show()

    layer_num_large = layer
    layer_err_large = layer_err

    np.savez('./training_stats_small.npz',
             loss_small=np.array(loss_small), loss_large=np.array(loss_large),
             layer_num_small=np.array(layer_num_small), layer_err_small=np.array(layer_err_small),
             layer_num_large=np.array(layer_num_large), layer_err_large=np.array(layer_err_large))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--dqn_config_file", type=str, default=None)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)

    args = parser.parse_args()

    config = make_config(args.config_file)
    logger = make_logger(config)

    if args.dqn_config_file is not None:
        dqn_config = make_config(args.dqn_config_file)
    else:
        dqn_config = None

    run_training_loop(config, logger, args, dqn_config)


if __name__ == "__main__":
    main()