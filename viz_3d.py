import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import skvideo.io
from tqdm import tqdm


def compute_mv_matrix(eye, angle):
    rot = Rotation.from_quat(angle).as_matrix()
    mv_matrix = np.concatenate([rot, eye[:, None]], axis=-1)
    mv_matrix = np.concatenate([mv_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    return mv_matrix


fname_prefix = sys.argv[1]
video = skvideo.io.vread(fname_prefix + '.mp4')
R = video.shape[1]
data = np.load(fname_prefix + '.npz')
T = video.shape[0]

all_points = []
all_colors = []

#viz = o3d.visualization.Visualizer()
#Gviz.create_window(visible=False)
all_frames = []
for t in tqdm(list(range(T))):
    eye = data['pos'][t]
    angle = data['rot'][t]
    mv_matrix = compute_mv_matrix(np.array([-4.371139e-09, 1.62, 0.05]), angle)
    mv_matrix = np.linalg.inv(data['mv_matrices'][t])
    
    p_matrix = data['proj_matrices'][t]
    depth_frame = data['depth'][t]
    rgb_frame = video[t] / 255.

    x = y = np.linspace(0, 1, R)
    coords = np.stack(np.meshgrid(x, y), axis=-1)
    coords = np.reshape(coords, (-1, 2))

    coords = 2 * coords - 1
    z = np.reshape(2 * depth_frame - 1, (-1, 1))

    clip_points = np.concatenate([coords, z, np.ones_like(z)], axis=-1)

    point = (np.linalg.inv(p_matrix) @ clip_points.T).T
    point = (mv_matrix @ point.T).T
    point /= point[:, [-1]]
    point = point[:, :-1]

    eye[1] *= -1
    point += eye

    colors = np.reshape(rgb_frame, (-1, 3))

    dists = np.linalg.norm(point - eye, axis=1)
    valid = []
    for i in range(len(dists)):
        if dists[i] > 50:
            continue
        valid.append(i)
    point, colors = point[valid], colors[valid]

    all_points.append(point)
    all_colors.append(colors)

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points))
    #pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))
    #viz.add_geometry(pcd)
    #img = (np.asarray(viz.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
    #all_frames.append(img)
    #viz.clear_geometries()

#all_frames = np.stack(all_frames)
#skvideo.io.vwrite('video.mp4', all_frames)

all_points = np.concatenate(all_points)
all_colors = np.concatenate(all_colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(all_colors)

o3d.visualization.draw_geometries([pcd])

#viz.add_geometry(pcd)

