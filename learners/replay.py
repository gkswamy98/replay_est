from imitation.algorithms import bc
from stable_baselines3.common import policies
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import argparse
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
from optim import OAdam
from envwrapper import BasicWrapper
from bc import *

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


class Replay():
    def __init__(self, base_env, eval_env, beta, mu):
        self.base_env = base_env
        self.replay_buffer = None
        self.beta = beta
        self.mu = mu
        self.first_alph_comp = True

        self.d3_obs = []
        self.d3_actions = []
        self.eval_env = eval_env
        self.bc_nets = []
        

    def train_ensemble(self, w, env, d1_size, bc_steps, load=False, num_trajs=None, n_ensemb=5):
        #This function runs step 2 and trains BC on D_1 to learn a query policy
        mean_rewards = []
        std_rewards = []

        expert_data_d1 = make_sa_dataloader(env, max_trajs=d1_size, normalize=False)
        
        for i in range(n_ensemb):
            if not load:
                mean_rewards = []
                std_rewards = []

                expert_data = make_sa_dataloader(env, max_trajs=num_trajs, normalize=False)
                test_env = gym.make(env)
                model = SAC('MlpPolicy', test_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), device="cuda:0")
                policy = BC(expert_data, model.policy.actor, steps=int(1e5))

                def get_policy(*args, **kwargs):
                    return policy
                model = PPO(get_policy, env, verbose=1)
                model.save(os.path.join("learners", env,
                                        "bc_query_{0}_{1}".format(num_trajs, i)))


            model = PPO.load(os.path.join("learners", env, "bc_query_{0}_{1}").format(num_trajs, i))
            mean_reward, std_reward = evaluate_policy(model, self.eval_env, n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("BC Net Reward: {0}".format(mean_rewards))
            print("loaded model")
            self.bc_nets.append(model)
        self.bc_nets = [x for y, x in sorted(zip(mean_rewards, self.bc_nets), reverse=True)]


    def generate_d3(self, env, target_samps, load, num_traj=None):
        if load:
            data = np.load("learners/{0}/d3_samps_{1}.npz".format(env, num_traj), allow_pickle=True)
            self.d3_obs = data["obs"]
            self.d3_actions = data["acts"]
            print("loaded d3", len(self.d3_obs))
        else:
            env_to_eval = self.base_env
            model = self.bc_nets[-1]

            cur_obs, cur_acts = [], []
            J = 0
            for _ in range(target_samps):
                obs = env_to_eval.reset()
                done = False
                while not done:
                    cur_obs.append(obs)
                    action, _ = model.predict(obs, state=None, deterministic=True)
                    cur_acts.append(action)
                    obs, reward, done, _ = env_to_eval.step(action)
                    J += reward

            self.d3_obs = cur_obs
            self.d3_actions = cur_acts
            mean_J = J / target_samps
            print(mean_J)
            np.savez("learners/{0}/d3_samps_{1}.npz".format(env, num_traj), obs = self.d3_obs, acts = self.d3_actions, mean_J = mean_J)

    def alpha_func(self, states):
        vals = []
        for mem in self.bc_nets:
            actions, _ = mem.predict(states)
            vals.append(actions)
        distance = np.linalg.norm(np.ptp(np.array(vals), axis=0), axis=1)

        return torch.sigmoid(torch.FloatTensor((self.mu - distance)/self.beta))


    def cost(self, f_function, d2_obs, d2_acts):
        sa_pairs_d3 = np.concatenate((self.d3_obs, self.d3_actions), axis=1)
        sa_pairs_d2 = np.concatenate((d2_obs, d2_acts), axis=1)

        self.first_exp_term = f_function.forward(torch.tensor(sa_pairs_d3, dtype=torch.float))
        self.sec_exp_term = f_function.forward(torch.tensor(sa_pairs_d2, dtype=torch.float))

        # only need to do this once
        if self.first_alph_comp:
            self.first_alpha_term = self.alpha_func(self.d3_obs)
            self.sec_alpha_term = 1 - self.alpha_func(d2_obs)

            combined = torch.mean(self.first_alpha_term) + torch.mean(self.sec_alpha_term)

            self.first_alpha_term = (1/combined) * self.first_alpha_term
            self.sec_alpha_term = (1/combined) * self.sec_alpha_term
            print('bc weight', torch.mean(self.first_alpha_term), 'exp weight', torch.mean(self.sec_alpha_term))
            self.first_alph_comp = False

        term_one = self.first_alpha_term * self.first_exp_term
        term_two = self.sec_alpha_term * self.sec_exp_term
        
        return torch.mean(term_one) + torch.mean(term_two)

    def sample_and_add(
        self,
        env,
        policy,
        trajs,
    ):
        #rollout trajectories using a policy and add to replay buffer
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

    def train(self, env, n_seed=0, n_exp=25):
        learn_rate = 8e-3
        outer_steps = 81
        inner_steps = 5000
        num_traj_sample = 4
        batch_size = 2048
        mean_rewards = []
        std_rewards = []

        pseudoreward = ReplayDiscriminator(self.base_env)
        pseudoreward_optimizer = OAdam(pseudoreward.parameters(), lr=learn_rate)

        #wrapped environment with modified reward -> -f function is the reward
        wrapped_env = BasicWrapper(self.base_env, pseudoreward)

        #initialize replay buffer
        replay_buffer = ReplayBuffer(wrapped_env.observation_space.shape[0], wrapped_env.action_space.shape[0])
        self.replay_buffer = replay_buffer

        
        model = SAC('MlpPolicy', wrapped_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                    learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02)

        #expert sa pairs for use in the gradient penalty
        expert_obs, expert_acts = make_sa_dataloader(env, max_trajs=n_exp, normalize=False, raw=True, raw_traj=False)
        expert_obs, expert_acts = np.array(expert_obs), np.array(expert_acts)
        expert_sa_pairs = torch.cat((torch.tensor(expert_obs), torch.tensor(expert_acts)), axis=1)

        #outer steps are maximizing the pseudo reward (f) function and inner steps are the minimization of policy over f
        for outer in range(outer_steps):
            if not outer == 0:
                learning_rate_used = learn_rate/outer
            else:
                learning_rate_used = learn_rate
            pseudoreward_optimizer = OAdam(pseudoreward.parameters(), lr=learning_rate_used)
            
            model.learn(total_timesteps=inner_steps, log_interval=5000)

            #sample from replay buffer
            low = wrapped_env.action_space.low
            high = wrapped_env.action_space.high
            tuple_samples = model.replay_buffer.sample(batch_size)
            obs_samples, act_samples = tuple_samples[0].cpu(), tuple_samples[1].cpu()
            act_samples = (((act_samples - low) / (high - low)) * 2.0) - 1.0
            sa_samples = torch.cat((torch.tensor(obs_samples), torch.tensor(act_samples)), axis=1)

            #Do the outer step - min C(f) - E_model(f)
            pseudoreward_optimizer.zero_grad()

            prog = outer/outer_steps

            c = self.cost(pseudoreward, self.d2_obs, self.d2_acts)
            learner_f_under_model = torch.mean(pseudoreward.forward(torch.tensor(sa_samples, dtype=torch.float)))

            random_sample = np.random.choice(
                len(expert_obs), len(obs_samples), replace=False)
            expert_sa_samples = expert_sa_pairs[random_sample]
            gp = gradient_penalty(sa_samples, expert_sa_samples, pseudoreward)

            #Maximize is same as minimize -(obj)
            obj = c - learner_f_under_model + 10 * gp
            obj.backward()

            pseudoreward_optimizer.step()

            #evaluate performance
            if outer % 5 == 0:
                mean_reward, std_reward = evaluate_policy(
                    model, self.eval_env, n_eval_episodes=10)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
            print("{0} Iteration: {1}".format(outer, mean_reward))
            np.savez(os.path.join("learners", env, "redo_replay_rewards_{0}_{1}_{2}".format(
                    n_exp, n_seed, outer)), means=mean_rewards, stds=std_rewards)


class ReplayBuffer():
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

        if not len(obs[0]) == self.obs_size or  not len(act[0]) == self.act_size:
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

class ReplayDiscriminator(nn.Module):
    def __init__(self, env):
        super(ReplayDiscriminator, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        net = create_mlp(self.obs_dim + self.action_dim, 1, self.net_arch, nn.ReLU)
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)

    def forward(self, inputs):
        output = self.net(inputs)
        return output.view(-1)
