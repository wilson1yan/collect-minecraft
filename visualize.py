import math
import os.path as osp
import glob
import argparse
import numpy as np
from moviepy.editor import ImageSequenceClip


def save_video_grid(video, fname=None, nrow=None, fps=10):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * ncol + padding,
                          (padding + w) * nrow + padding, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    if fname is not None:
        clip = ImageSequenceClip(list(video_grid), fps=fps)
        clip.write_gif(fname, fps=fps)
        print('saved videos to', fname)

    return video_grid # THWC, uint8


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, required=True)
parser.add_argument('-n', '--n_episodes', type=int, default=64, help='default: 64')
parser.add_argument('-f', '--fps', type=int, default=10)
args = parser.parse_args()

fnames = glob.glob(osp.join(args.data_path, '**', '*.npz'), recursive=True)
print(f'Found {len(fnames)} files')

fnames = np.random.choice(fnames, size=args.n_episodes, replace=False)
videos = []
for fname in fnames:
    data = np.load(fname, allow_pickle=True)
    videos.append(data['video'])
videos = np.stack(videos, axis=0)
save_video_grid(videos, fname='viz.gif', fps=args.fps)
