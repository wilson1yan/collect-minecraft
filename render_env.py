import cv2
import os
import os.path as osp
import multiprocessing as mp
import gym
import numpy as np
import argparse
from tqdm import tqdm
from env import SimpleExplore


class SimpleAgent:
    def __init__(self, prob_forward=0., action_repeat=5):
        self.action_repeat = action_repeat
        self.prob_forward = prob_forward
        self.counter = 0
        self.action = None

    def sample(self):
        if self.action is None or self.counter % self.action_repeat == 0:
            self.action = sample_action(self.prob_forward)
        self.counter += 1
        return self.action



def sample_action(prob_forward):
    prob_turn = (1 - prob_forward) / 2
    i = np.random.choice([0, 1, 2],
                         p=[prob_forward, prob_turn, prob_turn])
    if i == 0: # forward
        forward = jump = np.array(1)
        camera = np.array([0., 0.])
    elif i == 1: # left
        forward = jump = np.array(0)
        camera = np.array([0., -20.])
    elif i == 2: # right
        forward = jump = np.array(0)
        camera = np.array([0., 20.])
    else:
        raise ValueError('Invalid action', i)

    return dict(forward=forward, jump=jump, camera=camera), i


def main():
    abs_env = SimpleExplore()
    abs_env.register()

    env = gym.make('SimpleExplore-v0')
    agent = SimpleAgent(0.9, 5)

    env.reset()
    while True:
        env.render()
        action, _ = agent.sample()
        obs, _, _, _ = env.step(action)
        print([(o.shape, o.min(), o.max(), o.dtype) for o in obs['pov']])

if __name__ == '__main__':
    main()
