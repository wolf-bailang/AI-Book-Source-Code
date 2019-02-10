import numpy as np
import gym
from gym.spaces import Discrete

class SnakeEnv(gym.Env):
    SIZE=100
  
    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space=Discrete(self.SIZE+1)
        self.action_space=Discrete(len(dices))

        for k,v in self.ladders.items():
            self.ladders[v] = k
            # print 'ladders info:'
            # print self.ladders
            # print 'dice ranges:'
            # print self.dices
        self.pos = 1

    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, a):
        step = np.random.randint(1, self.dices[a] + 1)
        self.pos += step
        if self.pos == 100:
            return 100, 100, 1, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, -1, 0, {}

    def reward(self, s):
        if s == 100:
            return 100
        else:
            return -1

    def render(self):
        pass

# Q学习
class TableAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n    # 状态空间
        self.a_len = env.action_space.n         # 动作空间

        self.r = [env.reward(s) for s in range(0, self.s_len)]      # 回报 一维
        self.pi = np.array([0 for s in range(0, self.s_len)])       # 策略（|S|, |A|） 二维
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)     # 转移状态(|A|, |S|, |S|) 三维

        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)     # 矢量函数

        # enumerate() 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, dice in enumerate(env.dices):
            prob = 1.0 / dice
            for src in range(1, 100):
                step = np.arange(dice)
                step += src
                step = np.piecewise(step, [step > 100, step <= 100], [lambda x: 200 - x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob
        self.p[:, 100, 100]=1
        self.value_pi = np.zeros((self.s_len))
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state):
        return self.pi[state]

class ModelFreeAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.pi = np.array([0 for s in range(0, self.s_len)])
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.value_n = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state, epsilon = 0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_len)
        else:
            return self.pi[state]

# 策略评估
def eval_game(env, policy):
    state = env.reset()     # 获得状态
    return_val = 0
    while True:
        # isinstance判断一个对象是否是一个已知的类型
        if isinstance(policy, TableAgent) or isinstance(policy, ModelFreeAgent):
            act = policy.play(state)
        elif isinstance(policy, list):
            act = policy[state]
        else:
            raise Error('Illegal policy')
        state, reward, terminate, _ = env.step(act)
        # print state
        return_val += reward
        if terminate:
          break
    return return_val

if __name__ == '__main__':
    env = SnakeEnv(10, [3,6])
    env.reset()
    while True:
        state, reward, terminate, _ = env.step(0)
        print( reward, state)
        if terminate == 1:
            break





