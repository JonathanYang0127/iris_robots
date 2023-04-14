import numpy as np
import torch
import pickle
import argparse

from iris_robots.transformations import add_angles, angle_diff
import rlkit.torch.pytorch_util as ptu

ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/purple_marker_grasp_new/combined_trajectories.npy'
ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/purple_marker_grasp_franka/combined_trajectories.npy'
#MODEL_PATH = '/iris/u/jyang27/logs/22-11-28-BC-wx250/22-11-28-BC-wx250_2022_11_28_21_57_58_id000--s5/itr_1000.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-11-28-BC-wx250/22-11-28-BC-wx250_2022_11_28_21_58_14_id000--s6/itr_1000.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-11-29-BC-wx250/22-11-29-BC-wx250_2022_11_29_13_49_40_id000--s0/itr_1000.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-12-05-BC-wx250/22-12-05-BC-wx250_2022_12_05_23_38_55_id000--s2/itr_900.pt'
MODEL_PATH = '/iris/u/jyang27/logs/22-12-06-BC-wx250/22-12-06-BC-wx250_2022_12_06_10_28_11_id000--s0/itr_380.pt'

with open(ROBOT_PATH_ONE, 'rb') as f:
    traj1 = np.load(f, allow_pickle=True)

with open(ROBOT_PATH_TWO, 'rb') as f:
    traj2 = np.load(f, allow_pickle=True)

with open(MODEL_PATH, 'rb') as f:
    params = torch.load(f)
    policy = params['evaluation/policy']

def pose_diff(target, source):
    diff = np.zeros(len(target))
    diff[:3] = target[:3] - source[:3]
    diff[3:6] = angle_diff(target[3:6], source[3:6])
    diff[6] = target[6] - source[6]
    return diff


traj1_poses = []
for i in range(len(traj1)):
    closed_pose = None
    closed_idx = None
    for j in range(len(traj1[i]['actions'])):
        if traj1[i]['actions'][j][6] > 0.5:
            closed_pose = traj1[i]['observations'][j]['current_pose']
            closed_idx = j
            break
    
    pose_diffs = []
    for j in range(closed_idx):#len(traj1[i]['actions']) // 2):
        #d = traj1[i]['actions'][j]
        #d = pose_diff(traj1[i]['observations'][j]['current_pose'], closed_pose)
        d = pose_diff(traj1[i]['observations'][j + 1]['current_pose'], traj1[i]['observations'][j]['current_pose'])
        pose_diffs.append(d)

    traj1_poses.extend(pose_diffs)

for i in range(len(traj2)):
    for j in range(len(traj2[i]['actions'])):
        traj2[i]['actions'][j][3:6] *= -1
        traj2[i]['observations'][j]['current_pose'][3:6] *= -1

traj2_poses = []
for i in range(len(traj2)):
    closed_pose = None
    closed_idx = None
    for j in range(len(traj2[i]['actions'])):
        if traj2[i]['actions'][j][6] > 0.5:
            closed_pose = traj2[i]['observations'][j]['current_pose']
            closed_idx = j
            break

    pose_diffs = []
    for j in range(closed_idx):#len(traj2[i]['actions']) // 2):
        #d = traj2[i]['actions'][j]
        #d = pose_diff(traj2[i]['observations'][j]['current_pose'], closed_pose)
        d = pose_diff(traj2[i]['observations'][j + 1]['current_pose'], traj2[i]['observations'][j]['current_pose']) 
        pose_diffs.append(d)
    traj2_poses.extend(pose_diffs)

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

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
outputs_2d = pca.fit_transform(output)


from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
plt.scatter(outputs_2d[:NUM_POINTS, 0], outputs_2d[:NUM_POINTS, 1], c='r', label='WidowX Data')
plt.scatter(outputs_2d[NUM_POINTS:, 0], outputs_2d[NUM_POINTS:, 1], c='g', label='Franka Data')
plt.legend()

plt.savefig('out.png')



