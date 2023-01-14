from optim import OAdam
from envwrapper import BasicWrapper
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from utils import make_sa_dataloader, make_sads_dataloader, make_sa_dataset, linear_schedule, gradient_penalty
import os
from gym.spaces import Discrete
import gym
import pybullet_envs
from torch.nn import functional as F
import torch
from torch import nn
from typing import List, Type
from pybulletwrapper import OffsetWrapper
from noisywrapper import NoisyWrapper

def create_mlp(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    return modules

def init_ortho(layer):
    if type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight)

class WGAIL():
    def __init__(self, env):
        self.replay_buffer = None
        self.env = env

    def sample_and_add(
        self,
        env,
        policy,
        trajs,
    ):
        # rollout trajectories using a policy and add to replay buffer
        observations = []
        actions = []
        total_trajs = 0
        steps = 0
        while total_trajs < trajs:
            obs = env.reset()
            done = False
            while not done:
                observations.append(obs)
                act = policy.predict(obs)[0]
                actions.append(act)
                steps += 1
                obs, _, done, _ = env.step(act)
                if done:
                    total_trajs += 1
                    break
        self.replay_buffer.add(observations, actions)
        return steps

    def train(self, expert_sa_pairs, expert_obs, expert_acts, n_seed=0, n_exp=25):
        learn_rate = 8e-3 
        outer_steps = 81
        inner_steps = 5000
        save_inner_model = False
        num_traj_sample = 4
        batch_size = 2048
        save_rewards = True
        mean_rewards = []
        std_rewards = []

        cur_env = gym.make(self.env)
        cur_env = OffsetWrapper(cur_env)
        f_net = WGAILDiscriminator(cur_env)

        f_net_optimizer = OAdam(f_net.parameters(), lr=learn_rate)

        # wrapped environment with modified reward -> -f function is the reward
        wrapped_env = BasicWrapper(cur_env, f_net)

        # initialize replay buffer
        wgail_replay_buffer = WGAILReplayBuffer(
            wrapped_env.observation_space.shape[0], wrapped_env.action_space.shape[0])
        self.replay_buffer = wgail_replay_buffer

        # learner policy to optimize
        model = SAC('MlpPolicy', wrapped_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                    learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02, device="cuda:0")

        for outer in range(outer_steps):
            # update policy
            if not outer == 0:
                learning_rate_used = learn_rate/outer
            else:
                learning_rate_used = learn_rate
            f_net_optimizer = OAdam(f_net.parameters(), lr=learning_rate_used)
            
            model.learn(total_timesteps=inner_steps, log_interval=1000)

            if save_inner_model:
                model.save("sac_mimicmd_training_model")

            # sample from replay buffer
            low = wrapped_env.action_space.low
            high = wrapped_env.action_space.high
            tuple_samples = model.replay_buffer.sample(batch_size)
            obs_samples, act_samples = tuple_samples[0].cpu(), tuple_samples[1].cpu()
            act_samples = (((act_samples - low) / (high - low)) * 2.0) - 1.0
            sa_samples = torch.cat(
                (torch.tensor(obs_samples), torch.tensor(act_samples)), axis=1)

            # Do the outer step: min E_expert(f) - E_learner(f)
            f_net_optimizer.zero_grad()
            cost_value = torch.mean(f_net.forward(
                torch.tensor(expert_sa_pairs, dtype=torch.float)))
            learner_f_under_model = torch.mean(f_net.forward(
                torch.tensor(sa_samples, dtype=torch.float)))

            random_sample = np.random.choice(
                len(expert_obs), len(obs_samples), replace=False)
            new_expert_sa_pairs = torch.cat((torch.tensor(
                expert_obs[random_sample]), torch.tensor(expert_acts[random_sample])), axis=1)
            gp = gradient_penalty(sa_samples, new_expert_sa_pairs, f_net)

            # Maximize is same as minimize -(obj)
            obj = cost_value - learner_f_under_model + 10 * gp
            obj.backward()

            f_net_optimizer.step()

            # evaluate performance
            if outer % 5 == 0:
                mean_reward, std_reward = evaluate_policy(
                    model, OffsetWrapper(gym.make(self.env)), n_eval_episodes=10)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
            print("{0} Iteration: {1}".format(outer, mean_reward))
            if save_rewards:
                np.savez(os.path.join("learners", self.env, "off_mm_rewards_{0}_{1}_{2}".format(n_exp, n_seed,
                                                                                               outer)), means=mean_rewards, stds=std_rewards)


class WGAILReplayBuffer():
    def __init__(self, obs_space_size, action_space_size):
        self.obs_size = obs_space_size
        self.act_size = action_space_size
        self.size = 0
        self.obs = None
        self.actions = None
        self.first_addition = True

    def size():
        return self.size

    def add(self, obs, act):
        if not obs or not act:
            return

        if not len(obs[0]) == self.obs_size or not len(act[0]) == self.act_size:
            raise Exception('incoming samples do not match the correct size')
        if self.first_addition:
            self.first_addition = False
            self.obs = np.array(obs)
            self.actions = np.array(act)
        else:
            self.obs = np.append(self.obs, np.array(obs), axis=0)
            self.actions = np.append(self.actions, np.array(act), axis=0)
        self.size += len(obs)
        return

    def sample(self, batch):
        indexes = np.random.choice(range(self.size), batch)
        return self.obs[indexes], self.actions[indexes]


class WGAILDiscriminator(nn.Module):
    def __init__(self, env):
        super(WGAILDiscriminator, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        net = create_mlp(self.obs_dim + self.action_dim,
                         1, self.net_arch, nn.ReLU)
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)

    def forward(self, inputs):
        output = self.net(inputs)
        return output.view(-1)
