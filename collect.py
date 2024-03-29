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
    def __init__(self, prob_forward, action_repeat, max_consec_fwd):
        self.action_repeat = action_repeat
        self.prob_forward = prob_forward
        self.max_consec_fwd = max_consec_fwd
        self.reset()


    def reset(self):
        self.n_fwd = 0
        self.counter = 0
        self.action = None

    def sample(self):
        if self.n_fwd >= self.max_consec_fwd:
            prob_forward = 0.
        else:
            prob_forward = self.prob_forward

        if self.action is None or self.counter % self.action_repeat == 0:
            self.action = sample_action(prob_forward)

        if self.action[1] == 0:
            self.n_fwd += 1
        else:
            self.n_fwd = 0

        self.counter += 1
        return self.action


ACTIONS = {
    'forward': dict(forward=np.array(1), jump=np.array(1), camera=np.array([0., 0.])),
    'left': dict(forward=np.array(0), jump=np.array(1), camera=np.array([0., -20.])),
    'right': dict(forward=np.array(0), jump=np.array(1), camera=np.array([0., 20.])),
    'noop': dict(forward=np.array(0), jump=np.array(0), camera=np.array([0., 0.]))
}

ACTIONS_TO_ID = {
    'forward': 0,
    'left': 1,
    'right': 2,
}


def sample_action(prob_forward):
    prob_turn = (1 - prob_forward) / 2
    i = np.random.choice(['forward', 'left', 'right'],
                         p=[prob_forward, prob_turn, prob_turn])
    return ACTIONS[i], ACTIONS_TO_ID[i]

def collect_episode(env, agent, traj_length):
    agent.reset()
    obs = env.reset()
    observations, actions = [obs['pov']], [0]
    for t in range(traj_length):
        action, a_id = agent.sample()
        actions.append(a_id + 1)
        obs, _, done, _ = env.step(action)
        observations.append(obs['pov'])
        if done and t < traj_length - 1: # Invalid if the agent dies early
            return None

    rgb = np.stack(observations, axis=0)
    actions = np.array(actions, dtype=np.int32)
    return rgb, actions


def worker(id, args):
    args.output_dir = osp.join(args.output_dir, f'{id}')
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make('SimpleExplore-v0')
    agent = SimpleAgent(args.prob_forward, args.action_repeat, args.max_consec_fwd)

    num_episodes = args.num_episodes // args.n_parallel + (id < (args.num_episodes % args.n_parallel))
    pbar = tqdm(total=num_episodes, position=id)
    i = 0
    while i < num_episodes:
        out = collect_episode(env, agent, args.traj_length)
        if out is None:
            continue

        rgb, actions = out
        video_fname = osp.join(args.output_dir, f'{i:06d}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_fname, fourcc, 20.0, rgb.shape[1:-1])
        for t in range(rgb.shape[0]):
            frame = rgb[t]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        action_fname = osp.join(args.output_dir, f'{i:06d}.npz')
        np.savez_compressed(action_fname, actions=actions)
        i += 1
        pbar.update(1)
    pbar.close()


def main(args):
    abs_env = SimpleExplore(resolution=(args.resolution, args.resolution),
                            biomes=[6])
    abs_env.register()

    os.makedirs(args.output_dir, exist_ok=True)

    procs = [mp.Process(target=worker, args=(i, args)) for i in range(args.n_parallel)]
    [p.start() for p in procs]
    [p.join() for p in procs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-z', '--n_parallel', type=int, default=48,
                        help='default: 1')
    parser.add_argument('-a', '--action_repeat', type=int, default=5,
                        help='default: 5')
    parser.add_argument('-p', '--prob_forward', type=float, default=0.9,
                        help='default: 0.')
    parser.add_argument('-m', '--max_consec_fwd', type=int, default=50,
                        help='default: 25')
    parser.add_argument('-t', '--traj_length', type=int, default=300,
                        help='default: 100')
    parser.add_argument('-n', '--num_episodes', type=int, default=200000,
                        help='default: 100')
    parser.add_argument('-r', '--resolution', type=int, default=128,
                        help='default: 128')
    args = parser.parse_args()

    main(args)
