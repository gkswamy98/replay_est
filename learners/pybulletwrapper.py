import gym
import numpy as np

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
