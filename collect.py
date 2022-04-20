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

    
def collect_episode(env, agent, traj_length):
    observations, actions = [env.reset()], [0]
    observations.append(env.reset())
    for _ in range(traj_length):
        action, a_id = agent.sample()
        actions.append(a_id + 1)
        observations.append(env.step(action)[0])
    observations = np.stack(observations, axis=0) # THWC, uint8
    actions = np.array(actions, dtype=np.int32)
    return observations, actions

    
def worker(id, args):
    args.output_dir = osp.join(args.output_dir, f'{id}')
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make('SimpleExplore-v0', resolution=(args.resolution, args.resolution))
    agent = SimpleAgent(args.prob_forward, args.action_repeat)

    num_episodes = args.num_episodes // args.n_parallel + (id < (args.num_episodes % args.n_parallel))
    for i in tqdm(list(range(num_episodes)), position=id):
        observations, actions = collect_episode(env, agent, args.traj_length)
        fname = osp.join(args.output_dir, f'{i:06d}.npz')
        np.savez_compressed(fname, video=observations, actions=actions)
         
    
def main(args):
    abs_env = SimpleExplore()
    abs_env.register()

    os.makedirs(args.output_dir, exist_ok=True)
    
    procs = [mp.Process(target=worker, args=(i, args)) for i in range(args.n_parallel)]
    [p.start() for p in procs]
    [p.join() for p in procs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-z', '--n_parallel', type=int, default=1,
                        help='default: 1')
    parser.add_argument('-a', '--action_repeat', type=int, default=5,
                        help='default: 5')
    parser.add_argument('-p', '--prob_forward', type=float, default=0.,
                        help='default: 0.')
    parser.add_argument('-t', '--traj_length', type=int, default=100,
                        help='default: 100')
    parser.add_argument('-n', '--num_episodes', type=int, default=100,
                        help='default: 100')
    parser.add_argument('-r', '--resolution', type=int, default=128,
                        help='default: 128')
    args = parser.parse_args()

    main(args)