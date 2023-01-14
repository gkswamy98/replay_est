import gym
import argparse
import numpy as np
import os

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from utils import linear_schedule, make_vec_env
from torch import nn
import pybullet_envs
from stable_baselines3.common.vec_env import VecNormalize
from gym.spaces import Discrete

class NoisyWrapper(gym.Wrapper):
    def __init__(self, env, sigma=0.1): #0.5 for walker, 0.1 for hopper
        super().__init__(env)
        self.env = env
        self.sigma = sigma

    def step(self, action):
        action = action + np.random.normal(loc=0, scale=self.sigma)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class OffsetWrapper(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, base_env, sigma=1e-7):
        super(OffsetWrapper, self).__init__()
        self.base_env = base_env
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.sigma = sigma
    def reset(self):
        self.base_env.reset()
        pos, vel, joints = get_state(self.base_env)
        a = len(vel)
        b = len(vel[0])
        c = len(vel[0][0])
        u = [tuple([tuple(np.random.normal(scale=self.sigma, size=c)) for _ in range(b)]) for _ in range(a)]
        set_state(self.base_env, pos, u, joints)
        obs = self.base_env.robot.calc_state()
        return obs
    def step(self, action):
        return self.base_env.step(action)
    def render(self, mode='human'):
        self.base_env.render(mode=mode)
    def close (self):
        self.base_env.close()


def get_state(env):
    p = env.env._p
    base_pos = [] # position and orientation of base for each body
    base_vel = [] # velocity of base for each body
    joint_states = [] # joint states for each body
    for i in range(p.getNumBodies()):
        base_pos.append(p.getBasePositionAndOrientation(i))
        base_vel.append(p.getBaseVelocity(i))
        joint_states.append([p.getJointState(i,j) for j in range(p.getNumJoints(i))])
    return base_pos, base_vel, joint_states

def set_state(env, base_pos, base_vel, joint_states):
    p = env.env._p
    for i in range(p.getNumBodies()):
        p.resetBasePositionAndOrientation(i,*base_pos[i])
        p.resetBaseVelocity(i,*base_vel[i])
        for j in range(p.getNumJoints(i)):
            p.resetJointState(i,j,*joint_states[i][j][:2])
    

# All hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

def train_walker_expert():
    # No env normalization.
#     env = NoisyWrapper(gym.make('Walker2DBulletEnv-v0'))
#     model = SAC('MlpPolicy', env, verbose=1,
#                 buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
#                 train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=linear_schedule(7.3e-4), 
#                 learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
#                 use_sde=True)
#     model.learn(total_timesteps=1e6)
#     model.save("experts/Walker2DBulletEnv-v0/noisy_walker_expert")
    model = SAC.load("experts/Walker2DBulletEnv-v0/expert")
    gen_expert_demos('Walker2DBulletEnv-v0', gym.make('Walker2DBulletEnv-v0'), model, 25)


def train_hopper_expert():
    # No env normalization.
    env = gym.make('HopperBulletEnv-v0')
#     model = SAC('MlpPolicy', env, verbose=1,
#                 buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
#                 train_freq=8, gradient_steps=8, ent_coef='auto', learning_rate=linear_schedule(7.3e-4), 
#                 learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
#                 use_sde=True)
#     model.learn(total_timesteps=1e6)
#     model.save("experts/HopperBulletEnv-v0/noisy_hopper_expert5")
    model = SAC.load("experts/HopperBulletEnv-v0/expert")
    gen_expert_demos('HopperBulletEnv-v0', gym.make('HopperBulletEnv-v0'), model, 25)



def gen_expert_demos(dirname, env, model, num_trajs):
    trajs = dict()
    rewards = []
    ntraj = 0
    while ntraj < num_trajs:
        total_reward = 0
        obs = env.reset()
        done = False
        states = []
        actions = []
        while not done:
            states.append(obs)
            action, _state = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        if total_reward > 0:
            trajs[str(ntraj)] = {'states': np.array(
                states), 'actions': np.array(actions)}
            rewards.append(total_reward)
            ntraj += 1
            print(ntraj)
    print("Avg Reward:", np.mean(rewards))
    print(rewards)
    np.savez(os.path.join('experts', dirname, 'demos'), env=dirname,
             num_trajs=num_trajs,
             mean_reward=np.mean(rewards),
             std_reward=np.std(rewards),
             **trajs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train expert policies.')
    parser.add_argument('env', choices=['walker', 'hopper',])
    args = parser.parse_args()
    if args.env == 'walker':
        train_walker_expert()
    elif args.env == 'hopper':
        train_hopper_expert()
    else:
        print("ERROR: unsupported env.")
