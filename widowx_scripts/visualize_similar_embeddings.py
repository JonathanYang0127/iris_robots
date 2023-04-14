import numpy as np
import torch
import pickle
import argparse

import rlkit.torch.pytorch_util as ptu

ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_mixed_floral/combined_trajectories.npy'
ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_blue_floral/combined_trajectories.npy'
#MODEL_PATH = '/iris/u/jyang27/logs/22-12-01-BC-wx250/22-12-01-BC-wx250_2022_12_01_17_39_43_id000--s7/itr_580.pt'
MODEL_PATH = '/iris/u/jyang27/logs/23-01-07-BC-wx250/23-01-07-BC-wx250_2023_01_07_14_49_07_id000--s0/itr_800.pt'

with open(ROBOT_PATH_ONE, 'rb') as f:
    traj1 = np.load(f, allow_pickle=True)

with open(ROBOT_PATH_TWO, 'rb') as f:
    traj2 = np.load(f, allow_pickle=True)

with open(MODEL_PATH, 'rb') as f:
    params = torch.load(f)
    policy = params['evaluation/policy']

policy.output_conv_channels = False
policy.color_jitter = False
policy.feature_norm = False

def process_image(image, downsample=False):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    if downsample:
        image = image[:,::2, ::2]
    return image.flatten()


def process_obs(obs, task, use_robot_state, prev_obs=None, downsample=False):
    if use_robot_state:
        observation_keys = ['image', 'desired_pose', 'current_pose','joint_positions', 'joint_velocities', 'task_embedding']
    else:
        observation_keys = ['image', 'task_embedding']
    if prev_obs:
        observation_keys = ['previous_image'] + observation_keys
    if task is None:
        observation_keys = observation_keys[:-1]
    obs['image'] = process_image(obs['images'][0]['array'], downsample=downsample)  
    if prev_obs is not None:
        obs['previous_image'] = process_image(prev_obs['images'][0]['array'], downsample=downsample)
    obs['task_embedding'] = task
    return np.concatenate([obs[k] for k in observation_keys])

NUM_OBS = 1000
task = None
use_robot_state = False
downsample = True

obs1 = []
obs2 = []
num_traj1 = len(traj1)
num_traj2 = len(traj2)

for i in range(NUM_OBS):
    traj1_number = int(np.random.uniform(len(traj1)))
    traj2_number = int(np.random.uniform(len(traj2)))
    index1 = len(traj1[traj1_number]['observations']) // 2
    index2 = len(traj2[traj2_number]['observations']) // 2
    #index1 = int(np.random.uniform(len(traj1[traj1_number]['observations'])))
    #index2 = int(np.random.uniform(len(traj2[traj2_number]['observations'])))

    o1 = process_obs(traj1[traj1_number]['observations'][index1], np.array([0, 1]), use_robot_state, downsample=downsample)
    o2 = process_obs(traj2[traj2_number]['observations'][index2], np.array([1, 0]), use_robot_state, downsample=downsample)
    obs1.append(o1)
    obs2.append(o2)

obs1 = np.array(obs1)
obs2 = np.array(obs2)

from rlkit.torch.sac.policies import GaussianCNNPolicy, GaussianIMPALACNNPolicy, MakeDeterministic
output1 = super(GaussianCNNPolicy, policy).forward(ptu.from_numpy(obs1).cuda())
output2 = super(GaussianCNNPolicy, policy).forward(ptu.from_numpy(obs2).cuda())

import matplotlib.pyplot as plt
for image_index in range(20):
    image1 = obs1[image_index][:3 * 64 * 64].reshape(3, 64, 64).transpose(1, 2, 0)
    v1 = output1[image_index]
    closest = 1e9
    for v2 in output2:
        if torch.linalg.norm(v1 - v2) < closest:
            closest = torch.linalg.norm(v1 - v2)
            closest_image = obs2[image_index][:3 * 64 * 64].reshape(3, 64, 64).transpose(1, 2, 0)
    plt.imshow(image1)
    plt.savefig('images/out{}_original.png'.format(image_index))
    plt.imshow(closest_image)
    plt.savefig('images/out{}_nearest.png'.format(image_index))

output1 = output1.detach().cpu().numpy().reshape(NUM_OBS, -1)
output2 = output2.detach().cpu().numpy().reshape(NUM_OBS, -1)
output = np.concatenate((output1, output2))
print(output1.shape)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
outputs_2d = tsne.fit_transform(output)
'''

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
outputs_2d = pca.fit_transform(output)
'''

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
plt.scatter(outputs_2d[:NUM_OBS, 0], outputs_2d[:NUM_OBS, 1], c='r', label='Robot 1 Data')
plt.scatter(outputs_2d[NUM_OBS:, 0], outputs_2d[NUM_OBS:, 1], c='g', label='Robot 2 Data')
plt.legend()
plt.show()

plt.savefig('embeddings.png')


