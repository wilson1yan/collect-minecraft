import cv2
import os
import os.path as osp
import multiprocessing as mp
import gym
import numpy as np
import argparse
from tqdm import tqdm
from env import SimpleExplore
from scipy.spatial.transform import Rotation


class SimpleAgent:
    def __init__(self, prob_forward=0., action_repeat=5, max_consec_fwd=25, initial_sweep=False):
        self.action_repeat = action_repeat
        self.prob_forward = prob_forward
        self.max_consec_fwd = max_consec_fwd
        self.initial_sweep = initial_sweep
        self.reset()


    def reset(self):
        self.n_fwd = 0
        self.counter = 0
        self.action = None

    def sample(self):
        if self.initial_sweep and self.counter < 20:
            self.counter += 1
            return (ACTIONS['left'], ACTIONS_TO_ID['left'])

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
    'right': dict(forward=np.array(0), jump=np.array(1), camera=np.array([0., 20.]))
}

ACTIONS_TO_ID = {
    'forward': 0,
    'left': 1,
    'right': 2
}


def sample_action(prob_forward):
    prob_turn = (1 - prob_forward) / 2
    i = np.random.choice(['forward', 'left', 'right'], 
                         p=[prob_forward, prob_turn, prob_turn])
    return ACTIONS[i], ACTIONS_TO_ID[i]
    
def collect_episode(env, agent, traj_length):
    agent.reset()
    obs = env.reset()
    ls = obs['location_stats']
    pos = np.array([ls['xpos'], ls['ypos'], ls['zpos']], dtype=np.float32)
    rot = Rotation.from_matrix(np.linalg.inv(obs['pov'][2])[:3,:3]).as_quat()
    observations, actions = [(*obs['pov'], pos, rot)], [0]
    for t in range(traj_length):
        action, a_id = agent.sample()
        actions.append(a_id + 1)
        obs, _, done, _ = env.step(action)
        ls = obs['location_stats']
        pos = np.array([ls['xpos'], ls['ypos'], ls['zpos']], dtype=np.float32)
        rot = Rotation.from_matrix(np.linalg.inv(obs['pov'][2])[:3,:3])
        rot = rot.as_quat()
        observations.append((*obs['pov'], pos, rot))
        if done and t < traj_length - 1:
            return None

    if args.rgb_only:
        observations = np.stack(observations, axis=0) # THWC, uint8
    else:
        rgb, depth, mv, proj, pos, rot = [np.stack(x,axis=0) for x in zip(*observations)]
        observations = (rgb, depth, mv, proj, pos, rot)
    
    actions = np.array(actions, dtype=np.int32)
    return observations, actions

    
def worker(id, args):
    args.output_dir = osp.join(args.output_dir, f'{id}')
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make('SimpleExplore-v0')
    agent = SimpleAgent(args.prob_forward, args.action_repeat, args.max_consec_fwd, args.initial_sweep)

    num_episodes = args.num_episodes // args.n_parallel + (id < (args.num_episodes % args.n_parallel))
    pbar = tqdm(total=num_episodes, position=id)
    i = 0
    while i < num_episodes:
        out = collect_episode(env, agent, args.traj_length)
        if out is None:
            continue

        observations, actions = out

        rgb = observations if args.rgb_only else observations[0]
        video_fname = osp.join(args.output_dir, f'{i:06d}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_fname, fourcc, 20.0, rgb.shape[1:-1])
        for t in range(rgb.shape[0]):
            frame = rgb[t]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        if args.rgb_only:
            action_fname = osp.join(args.output_dir, f'{i:06d}.npy')
            np.save(action_fname, actions)
        else:
            other_fname = osp.join(args.output_dir, f'{i:06d}.npz')
            depth, mv, proj, pos, rot = observations[1:]
            
            # Modelview matrix to pose
            def _mv_to_pose(mv):
                mv = np.linalg.inv(mv)
                rot, pos = mv[:3, :3], mv[:3, -1]
                pos = pos.astype(np.float32)
                rot = Rotation.from_matrix(rot).as_quat().astype(np.float32)
                return pos, rot
            pose = [_mv_to_pose(mv[t]) for t in range(mv.shape[0])]
            _, _ = [np.stack(x, axis=0) for x in zip(*pose)]
            
            np.savez_compressed(other_fname, actions=actions, depth=depth,
                                proj_matrices=proj, mv_matrices=mv, pos=pos, rot=rot)
        i += 1
        pbar.update(1)
    pbar.close()
         
    
def main(args):
    abs_env = SimpleExplore(resolution=(args.resolution, args.resolution), 
                            include_depth=not args.rgb_only, biomes=[134])#biomes=[140, 38, 158, 133, 4, 27, 134, 8, 37, 165, 38, 166, 13, 17, 18, 19, 31])
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
    parser.add_argument('-m', '--max_consec_fwd', type=int, default=25,
                        help='default: 25')
    parser.add_argument('-s', '--initial_sweep', action='store_true')
    parser.add_argument('-t', '--traj_length', type=int, default=100,
                        help='default: 100')
    parser.add_argument('-n', '--num_episodes', type=int, default=100,
                        help='default: 100')
    parser.add_argument('-r', '--resolution', type=int, default=128,
                        help='default: 128')
    parser.add_argument('--rgb_only', action='store_true')
    args = parser.parse_args()

    main(args)
