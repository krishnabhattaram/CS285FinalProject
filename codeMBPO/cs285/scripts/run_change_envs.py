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


def collect_mbpo_rollout(
        env: gym.Env,
        dos_model: DOSModel,
        dqn_agent: DQNAgent,
        ob: np.ndarray,
        rollout_len: int = 1,
):
    obs, acs, rewards, next_obs, dones = [], [], [], [], []
    for _ in range(rollout_len):
        # TODO(student): collect a rollout using the learned dynamics models
        # HINT: get actions from `sac_agent` and `next_ob` predictions from `mb_agent`.
        # Average the ensemble predictions directly to get the next observation.
        # Get the reward using `env.get_reward`.

        ac = dqn_agent.get_action(ob)
        next_ob = env.get_next_observation(ob, ac)
        next_dos = dos_model.get_dos_ensemble_prediction(next_ob[None, :])
        rew = env.get_reward(next_dos, next_dos)

        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        dones.append(False)

        ob = next_ob

    return {
        "observation": np.array(obs),
        "action": np.array(acs),
        "reward": np.array(rewards),
        "next_observation": np.array(next_obs),
        "done": np.array(dones),
    }


def run_training_loop(
        config: dict, logger: Logger, args: argparse.Namespace, dqn_config: Optional[dict]
):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    small_env = gym.make('NanoslabSiCachedEnvSmall-v0', render_mode=None)
    large_env = gym.make('NanoslabSiCachedEnvLarge-v0', render_mode=None)

    small_eval_env = gym.make('NanoslabSiCachedEnvSmall-v0', render_mode=None)
    large_eval_env = gym.make('NanoslabSiCachedEnvLarge-v0', render_mode=None)

    ep_len = config["ep_len"] or small_env.spec.max_episode_steps

    # initialize agent
    dos_model = DOSModel(
        small_env,
        num_dos=200,
        **config["agent_kwargs"],
    )

    print(f'DOS Model has {sum([np.prod(p.size()) for p in dos_model.parameters()])} parameters')

    small_replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])
    large_replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    # # if doing MBPO, initialize SAC and make that our main agent that we use to
    # # collect data and evaluate
    # if dqn_config is not None:
    dqn_agent = DQNAgent(
        small_eval_env.observation_space.shape,
        small_env.action_space.n,
        **dqn_config["agent_kwargs"],
    )

    small_dqn_replay_buffer = ReplayBuffer(dqn_config["replay_buffer_capacity"])
    large_dqn_replay_buffer = ReplayBuffer(dqn_config["replay_buffer_capacity"])

    actor_agent = dqn_agent

    total_envsteps = 0

    for itr in tqdm.tqdm(range(config["num_iters"])):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")
        if itr == 0:
            # TODO(student): collect at least config["initial_batch_size"] transitions with a random policy
            # HINT: Use `utils.RandomPolicy` and `utils.sample_trajectories`
            trajs, envsteps_this_batch = utils.sample_trajectories(env=small_env,
                                                                   policy=utils.RandomPolicy(env=small_env),
                                                                   min_timesteps_per_batch=config["initial_batch_size"],
                                                                   max_length=ep_len,
                                                                   render=False)
        else:
            # TODO(student): collect at least config["batch_size"] transitions with our `actor_agent`
            trajs, envsteps_this_batch = utils.sample_trajectories(env=small_env,
                                                                   policy=actor_agent,
                                                                   min_timesteps_per_batch=config["batch_size"],
                                                                   max_length=ep_len,
                                                                   render=False)

        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        for traj in trajs:
            small_replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # # if doing MBPO, add the collected data to the DQN replay buffer as well
        # if dqn_config is not None:
        for traj in trajs:
            small_dqn_replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # update agent's statistics with the entire replay buffer
        dos_model.update_statistics(
            obs=small_replay_buffer.observations[: len(small_replay_buffer)],
            dos=small_env.get_dos(small_replay_buffer.observations[: len(small_replay_buffer)]),
        )

        # train agent
        print("Training agent...")
        all_losses = []
        for _ in tqdm.trange(
                config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            step_losses = []
            # TODO(student): train the dynamics models
            # HINT: train each dynamics model in the ensemble with a *different* batch of transitions!
            # Use `replay_buffer.sample` with config["train_batch_size"].
            if isinstance(dos_model, DOSModelSchnet):
                transitions = small_replay_buffer.sample(config["train_batch_size"])
                step_losses += [dos_model.update(i=0, obs=transitions["observations"],
                                                 dos=small_env.get_dos(transitions["observations"]))]
            else:
                for i in range(dos_model.ensemble_size):
                    transitions = small_replay_buffer.sample(config["train_batch_size"])
                    step_losses += [dos_model.update(i=i, obs=transitions["observations"],
                                                     dos=small_env.get_dos(transitions["observations"]))]

            all_losses.append(np.mean(step_losses))

        # # on iteration 0, plot the full learning curve
        # if itr > -1:
        #     plt.figure()
        #     plt.plot(all_losses)
        #     plt.title(f"Iteration {itr}: Dynamics Model Training Loss")
        #     plt.ylabel("Loss")
        #     plt.xlabel("Step")
        #     plt.savefig(os.path.join(logger._log_dir, "itr_0_loss_curve.png"))
        #
        #     obs = small_env.reset()[None, :]
        #     dos_obs = small_env.get_dos(obs).squeeze()
        #     dos_pred = dos_model.get_dos_ensemble_prediction(obs).squeeze()
        #     plt.figure()
        #     plt.plot(dos_obs, label='Actual')
        #     plt.plot(dos_pred, label='Predicted')
        #     plt.legend()

        # log the average loss
        loss = np.mean(all_losses)
        logger.log_scalar(loss, "dynamics_loss", itr)

        # # for MBPO: now we need to train the SAC agent
        # if dqn_config is not None:
        print("Training SAC agent...")
        for i in tqdm.trange(
                dqn_config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            if dqn_config["mbpo_rollout_length"] > 0:
                # collect a rollout using the dynamics model
                rollout = collect_mbpo_rollout(
                    small_env,
                    dos_model,
                    dqn_agent,
                    # sample one observation from the "real" replay buffer
                    small_replay_buffer.sample(1)["observations"][0],
                    dqn_config["mbpo_rollout_length"],
                )
                # insert it into the DQN replay buffer only
                small_dqn_replay_buffer.batched_insert(
                    observations=rollout["observation"],
                    actions=rollout["action"],
                    rewards=rollout["reward"],
                    next_observations=rollout["next_observation"],
                    dones=rollout["done"],
                )
            # train SAC
            batch = small_dqn_replay_buffer.sample(dqn_config["batch_size"])
            dqn_agent.update(
                ptu.from_numpy(batch["observations"]),
                ptu.from_numpy(batch["actions"]),
                ptu.from_numpy(batch["rewards"]),
                ptu.from_numpy(batch["next_observations"]),
                ptu.from_numpy(batch["dones"]),
                i,
            )

        # Run evaluation
        if config["num_eval_trajectories"] == 0:
            continue
        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        trajs = utils.sample_n_trajectories(
            small_eval_env,
            policy=actor_agent,
            ntraj=config["num_eval_trajectories"],
            max_length=ep_len,
        )
        returns = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]

        logger.log_scalar(np.mean(returns), "eval_return", itr)
        logger.log_scalar(np.mean(ep_lens), "eval_ep_len", itr)
        print(f"Average eval return: {np.mean(returns)}")

        if len(returns) > 1:
            logger.log_scalar(np.std(returns), "eval/return_std", itr)
            logger.log_scalar(np.max(returns), "eval/return_max", itr)
            logger.log_scalar(np.min(returns), "eval/return_min", itr)
            logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", itr)
            logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", itr)
            logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", itr)



    ### Switching Environments!!!
    dos_model.env = large_env
    total_envsteps = 0

    print('\n\n\nStarting Large System!')
    for itr in tqdm.tqdm(range(config["num_iters"], 2*config["num_iters"])):
        print(f"\n\n********** Iteration {itr} ************")
        # collect data
        print("Collecting data...")

        trajs, envsteps_this_batch = utils.sample_trajectories(env=large_env,
                                                                   policy=actor_agent,
                                                                   min_timesteps_per_batch=config["batch_size"],
                                                                   max_length=ep_len,
                                                                   render=False)

        total_envsteps += envsteps_this_batch
        logger.log_scalar(total_envsteps, "total_envsteps", itr)

        # insert newly collected data into replay buffer
        for traj in trajs:
            large_replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # # if doing MBPO, add the collected data to the DQN replay buffer as well
        # if dqn_config is not None:
        for traj in trajs:
            large_dqn_replay_buffer.batched_insert(
                observations=traj["observation"],
                actions=traj["action"],
                rewards=traj["reward"],
                next_observations=traj["next_observation"],
                dones=traj["done"],
            )

        # update agent's statistics with the entire replay buffer
        dos_model.update_statistics(
            obs=large_replay_buffer.observations[: len(large_replay_buffer)],
            dos=large_env.get_dos(large_replay_buffer.observations[: len(large_replay_buffer)]),
        )

        # Train agent in new environment
        print("Training agent...")
        all_losses = []
        for _ in tqdm.trange(
                config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            step_losses = []
            # TODO(student): train the dynamics models
            # HINT: train each dynamics model in the ensemble with a *different* batch of transitions!
            # Use `replay_buffer.sample` with config["train_batch_size"].
            if isinstance(dos_model, DOSModelSchnet):
                transitions = large_replay_buffer.sample(config["train_batch_size"])
                step_losses += [dos_model.update(i=0, obs=transitions["observations"],
                                                 dos=large_env.get_dos(transitions["observations"]))]
            else:
                for i in range(dos_model.ensemble_size):
                    transitions = large_replay_buffer.sample(config["train_batch_size"])
                    step_losses += [dos_model.update(i=i, obs=transitions["observations"],
                                                     dos=large_env.get_dos(transitions["observations"]))]

            all_losses.append(np.mean(step_losses))

        # # on iteration 0, plot the full learning curve
        # plt.figure()
        # plt.plot(all_losses)
        # plt.title(f"Iteration Large {itr}: Dynamics Model Training Loss")
        # plt.ylabel("Loss")
        # plt.xlabel("Step")

        obs = large_env.reset()[None, :]
        dos_obs = large_env.get_dos(obs).squeeze()
        dos_pred = dos_model.get_dos_ensemble_prediction(obs).squeeze()
        plt.figure()
        plt.plot(dos_obs, label='Large Actual')
        plt.plot(dos_pred, label='Large Predicted')
        plt.legend()
        plt.show()

        # log the average loss
        loss = np.mean(all_losses)
        logger.log_scalar(loss, "dynamics_loss", itr)

        # # for MBPO: now we need to train the SAC agent
        # if dqn_config is not None:
        print("Training SAC agent...")
        for i in tqdm.trange(
                dqn_config["num_agent_train_steps_per_iter"], dynamic_ncols=True
        ):
            if dqn_config["mbpo_rollout_length"] > 0:
                # collect a rollout using the dynamics model
                rollout = collect_mbpo_rollout(
                    large_env,
                    dos_model,
                    dqn_agent,
                    # sample one observation from the "real" replay buffer
                    large_replay_buffer.sample(1)["observations"][0],
                    dqn_config["mbpo_rollout_length"],
                )
                # insert it into the DQN replay buffer only
                large_dqn_replay_buffer.batched_insert(
                    observations=rollout["observation"],
                    actions=rollout["action"],
                    rewards=rollout["reward"],
                    next_observations=rollout["next_observation"],
                    dones=rollout["done"],
                )
            # train SAC
            batch = large_dqn_replay_buffer.sample(dqn_config["batch_size"])
            dqn_agent.update(
                ptu.from_numpy(batch["observations"]),
                ptu.from_numpy(batch["actions"]),
                ptu.from_numpy(batch["rewards"]),
                ptu.from_numpy(batch["next_observations"]),
                ptu.from_numpy(batch["dones"]),
                i,
            )

        # Run evaluation
        if config["num_eval_trajectories"] == 0:
            continue
        print(f"Evaluating {config['num_eval_trajectories']} rollouts...")
        trajs = utils.sample_n_trajectories(
            large_eval_env,
            policy=actor_agent,
            ntraj=config["num_eval_trajectories"],
            max_length=ep_len,
        )
        returns = [t["episode_statistics"]["r"] for t in trajs]
        ep_lens = [t["episode_statistics"]["l"] for t in trajs]

        logger.log_scalar(np.mean(returns), "eval_return", itr)
        logger.log_scalar(np.mean(ep_lens), "eval_ep_len", itr)
        print(f"Average eval return: {np.mean(returns)}")

        if len(returns) > 1:
            logger.log_scalar(np.std(returns), "eval/return_std", itr)
            logger.log_scalar(np.max(returns), "eval/return_max", itr)
            logger.log_scalar(np.min(returns), "eval/return_min", itr)
            logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", itr)
            logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", itr)
            logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", itr)



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
