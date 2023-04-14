import numpy as np
import torch
import pickle
import argparse

from iris_robots.transformations import add_angles, angle_diff
import rlkit.torch.pytorch_util as ptu
from pathlib import Path

ROBOT_PATHS = ['/iris/u/jyang27/training_data/franka_nodesired/',
    '/iris/u/jyang27/training_data/wx250_nodesired_control3/']


#MODEL_PATH = '/iris/u/jyang27/logs/22-11-28-BC-wx250/22-11-28-BC-wx250_2022_11_28_21_57_58_id000--s5/itr_1000.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-11-28-BC-wx250/22-11-28-BC-wx250_2022_11_28_21_58_14_id000--s6/itr_1000.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-11-29-BC-wx250/22-11-29-BC-wx250_2022_11_29_13_49_40_id000--s0/itr_1000.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-12-05-BC-wx250/22-12-05-BC-wx250_2022_12_05_23_38_55_id000--s2/itr_900.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-12-06-BC-wx250/22-12-06-BC-wx250_2022_12_06_10_28_11_id000--s0/itr_380.pt'

buffers = set()
for buffer_path in ROBOT_PATHS:
    if '.pkl' in buffer_path or '.npy' in buffer_path:
        buffers.add(buffer_path)
    else:
        path = Path(buffer_path)
        buffers.update(list(path.rglob('combined_trajectories.npy')))
buffers = [str(b) for b in buffers]
print(buffers)


traj = []
for b in buffers:
    with open(b, 'rb') as f:
        rollouts = np.load(f, allow_pickle=True)
    traj.extend(rollouts)


#with open(MODEL_PATH, 'rb') as f:
#    params = torch.load(f)
#    policy = params['evaluation/policy']

def pose_diff(target, source):
    diff = np.zeros(len(target))
    diff[:3] = target[:3] - source[:3]
    diff[3:6] = angle_diff(target[3:6], source[3:6])
    diff[6] = target[6] - source[6]
    return diff


traj_poses = []
traj_images = []
for i in range(len(traj)):
    closed_pose = None
    closed_idx = None
    for j in range(len(traj[i]['actions'])):
        if traj[i]['actions'][j][6] > 0.5:
            closed_pose = traj[i]['observations'][j]['current_pose']
            closed_idx = j
            break
    
    pose_diffs = []
    for j in range(len(traj[i]['actions']) - 1):
        #d = traj1[i]['actions'][j]
        #d = pose_diff(traj1[i]['observations'][j]['current_pose'], closed_pose)
        d = pose_diff(traj[i]['observations'][j + 1]['current_pose'], traj[i]['observations'][closed_idx]['current_pose'])
        pose_diffs.append(d)
        traj_images.append(traj[i]['observations'][j]['images'][0]['array'].astype(np.float32) / 255.0)

    traj_poses.extend(pose_diffs)


traj_poses = np.array(traj_poses)
dists = np.zeros((traj_poses.shape[0]))
traj_idx = 30
num_choices = 50
for i in range(len(traj_poses)):
    dist = np.linalg.norm(traj_poses[i] - traj_poses[traj_idx]) 
    dists[i] = dist

closest_idxs = np.argpartition(dists, num_choices)[:num_choices]
print(closest_idxs)
idx = 0
img1 = traj_images[traj_idx]


import matplotlib.pyplot as plt
plt.axis('off')
for i in range(len(closest_idxs)):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img1)
    ax[1].imshow(traj_images[closest_idxs[i]])
    plt.savefig('closest_images/nearest_pose_{}.png'.format(i))
    plt.close(fig)
'''
NUM_POINTS = 1000
traj1_poses = np.array(traj1_poses)
traj2_poses = np.array(traj2_poses)
print(traj1_poses.shape)
traj1_indices = np.random.randint(len(traj1_poses), size=NUM_POINTS)
traj2_indices = np.random.randint(len(traj2_poses), size=NUM_POINTS)
output = np.concatenate((traj1_poses[traj1_indices], traj2_poses[traj2_indices]))


print(output.shape)
#from sklearn.manifold import TSNE
#tsne = TSNE(n_components=2, random_state=0)
#outputs_2d = tsne.fit_transform(output)

#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#outputs_2d = pca.fit_transform(output)


from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
#plt.scatter(outputs_2d[:NUM_POINTS, 0], outputs_2d[:NUM_POINTS, 1], c='r', label='WidowX Data')
#plt.scatter(outputs_2d[NUM_POINTS:, 0], outputs_2d[NUM_POINTS:, 1], c='g', label='Franka Data')
#plt.legend()

plt.savefig('out.png')
'''


