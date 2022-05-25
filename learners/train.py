from imitation.algorithms import adversarial, bc
from imitation.util import logger, util
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common import policies
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import argparse
from utils import make_sa_dataloader, make_sads_dataloader, make_sa_dataset
import os
import gym
import pybullet_envs
from pybulletwrapper import OffsetWrapper
from wgail import *
from replay import *

def train_bc(env, n=0):
    venv = util.make_vec_env(env, n_envs=8)
    w = 256
    for i in range(n):
        mean_rewards = []
        std_rewards = []
        for num_trajs in [6, 12, 18]:
            if num_trajs == 0:
                expert_data = make_sa_dataloader(env, normalize=False)
            else:
                expert_data = make_sa_dataloader(env, max_trajs=num_trajs, normalize=False)
            bc_trainer = bc.BC(observation_space=venv.observation_space, action_space=venv.action_space, expert_data=expert_data,
                               policy_class=policies.ActorCriticPolicy,
                               ent_weight=0., l2_weight=0., policy_kwargs=dict(net_arch=[w, w]))
            if num_trajs > 0:
                bc_trainer.train(n_epochs=int(1e5/100))

            def get_policy(*args, **kwargs):
                return bc_trainer.policy
            model = PPO(get_policy, env, verbose=1)
            model.save(os.path.join("learners", env,
                                    "bc_{0}_{1}".format(i, num_trajs)))
            test_env = gym.make(env)
            test_env = OffsetWrapper(test_env)
            mean_reward, std_reward = evaluate_policy(
                    model, test_env, n_eval_episodes=10)               
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Trajs: {1}".format(num_trajs, mean_reward))
            np.savez(os.path.join("learners", env, "bc_rewards_{0}".format(
                i)), means=mean_rewards, stds=std_rewards)

def train_wgail(env, n=0):
    for i in range(n):
        for num_trajs in [6, 12, 18]:
            expert_obs, expert_acts = make_sa_dataloader(env, normalize=False, max_trajs=num_trajs, raw=True, raw_traj=False)
            expert_obs, expert_acts = np.array(expert_obs), np.array(expert_acts)
            expert_sa_pairs = torch.cat((torch.tensor(expert_obs), torch.tensor(expert_acts)), axis=1)
            wgail_instance = WGAIL(env)
            wgail_instance.train(expert_sa_pairs, expert_obs, expert_acts, n_seed=i, n_exp=num_trajs)

def train_replay(env, n=0):
    for i in range(n):
        for num_traj in [6, 12, 18]:
            train_rep(env, num_traj=num_traj, seed=i)

def train_rep(env, n=0, bc_steps=0, num_traj=25, seed=0):
    beta = 0.1
    mu = 1.0

    replay_instance = Replay(OffsetWrapper(gym.make(env)), OffsetWrapper(gym.make(env)), beta, mu)
    action_space_size = gym.make(env).action_space.shape[0]
    obs_space_size = gym.make(env).observation_space.shape[0]

    # 1. Generate D1 and D2.
    ratio_d1_d2 = 2.0/num_traj
    traj_numbers = list(range(num_traj))
    d1_size = traj_numbers[:int(num_traj*ratio_d1_d2)]
    d2_size = traj_numbers[int(num_traj*ratio_d1_d2):]

    d1_obs, d1_acts = make_sa_dataloader(env, max_trajs=d1_size, normalize=False, raw=True, raw_traj=False)
    d2_obs, d2_acts = make_sa_dataloader(env, max_trajs=d2_size, normalize=False, raw=True, raw_traj=False)
    replay_instance.d1_obs = d1_obs
    replay_instance.d1_acts = d1_acts
    replay_instance.d2_obs = d2_obs
    replay_instance.d2_acts = d2_acts
    print(d1_size, d2_size)

    # 2. Generate D3.
    #Choose to use existing query nets or train new ones
    load_query = False
    bc_steps = int(1e3)
    replay_instance.train_ensemble(256, env, d1_size, bc_steps, load=load_query, num_trajs=len(d1_size))

    #Choose to use existing generated bc data or create [target_samps] new trajectories
    target_samps = 100
    load_d3 = False
    replay_instance.generate_d3(env, target_samps, load_d3, num_traj=num_traj)

    # 3. Match weighted moments.
    replay_instance.train(env, n_exp=num_traj, n_seed=seed)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train learner policies.')
    parser.add_argument(
        '-a', '--algo', choices=['bc', 'wgail', 'replay',], required=True)
    parser.add_argument('-e', '--env', choices=['walker', 'hopper'],
                        required=True)
    parser.add_argument('-n', '--num_runs', required=False)
    args = parser.parse_args()
    if args.env == "walker":
        envname = 'Walker2DBulletEnv-v0'
    elif args.env == "hopper":
        envname = 'HopperBulletEnv-v0'
    else:
        print("ERROR: unsupported env.")
    if args.num_runs is not None and args.num_runs.isdigit():
        num_runs = int(args.num_runs)
    else:
        num_runs = 1
    if args.algo == 'bc':
        train_bc(envname, num_runs)
    elif args.algo == 'wgail':
        train_wgail(envname, num_runs)
    elif args.algo == "replay":
        train_replay(envname, num_runs)
    else:
        print("ERROR: unsupported algorithm")
