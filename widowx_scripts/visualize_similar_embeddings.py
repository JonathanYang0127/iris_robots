import numpy as np
import torch
import pickle
import argparse

import rlkit.torch.pytorch_util as ptu

#ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_blue_nodesired_control3/combined_trajectories.npy'
ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_tan_nodesired_control3/combined_trajectories.npy'
ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_gray_red_cup_nodesired_control3/combined_trajectories.npy'
#ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/franka_nodesired/franka_black_marker_grasp_gray_nodesired/combined_trajectories.npy'
#ROBOT_PATH_TWO = ROBOT_PATH_ONE
#ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_blue_nodesired_control3/combined_trajectories.npy'
#MODEL_PATH = '/iris/u/jyang27/logs/23-02-21-BC-wx250/23-02-21-BC-wx250_2023_02_21_22_26_05_id000--s2/itr_360.pt' 
#MODEL_PATH = '/iris/u/jyang27/logs/23-02-22-BC-wx250/23-02-22-BC-wx250_2023_02_22_08_36_51_id000--s2/itr_200.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/23-02-22-BC-wx250/23-02-22-BC-wx250_2023_02_22_15_58_09_id000--s2/itr_120.pt'
MODEL_PATH = '/iris/u/jyang27/logs/23-02-22-BC-wx250/23-02-22-BC-wx250_2023_02_22_15_58_09_id000--s2/itr_1300.pt'

with open(ROBOT_PATH_ONE, 'rb') as f:
    traj1 = np.load(f, allow_pickle=True)

with open(ROBOT_PATH_TWO, 'rb') as f:
    traj2 = np.load(f, allow_pickle=True)

with open(MODEL_PATH, 'rb') as f:
    params = torch.load(f)
    policy = params['evaluation/policy']

#policy.output_conv_channels = False
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
        observation_keys = ['image', 'desired_pose', 'current_pose', 'task_embedding']
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

NUM_OBS = 50
task = None
use_robot_state = True
downsample = True

obs1 = []
obs2 = []
num_traj1 = len(traj1)
num_traj2 = len(traj2)

for i in range(len(traj1)):
    for j in range(len(traj1[i]['observations'])):
        o1 = process_obs(traj1[i]['observations'][j], np.array([0, 0]), use_robot_state, downsample=downsample)
        obs1.append(o1)


for i in range(len(traj2)):
    for j in range(len(traj2[i]['observations'])):
        o2 = process_obs(traj2[i]['observations'][j], np.array([0, 0]), use_robot_state, downsample=downsample)
        obs2.append(o2)

obs1 = np.array(obs1)
obs2 = np.array(obs2)

output1, output2 = [], []
from rlkit.torch.sac.policies import GaussianCNNPolicy, GaussianIMPALACNNPolicy, MakeDeterministic
for i in range(len(obs1)):
    img = ptu.from_numpy(np.array([obs1[i]])).cuda()
    _, e1 = policy.forward(img, intermediate_output_layer=2)
    output1.append(e1[0].detach().cpu().numpy())

for i in range(len(obs2)):
    img = np.array([obs2[i]])
    _, e2 = policy.forward(ptu.from_numpy(img).cuda(), intermediate_output_layer=2)
    output2.append(e2[0].detach().cpu().numpy())

import matplotlib.pyplot as plt
for i in range(20):
    image_index = np.random.randint(len(obs1))
    image1 = obs1[image_index][:3 * 64 * 64].reshape(3, 64, 64).transpose(1, 2, 0)
    v1 = output1[image_index]
    closest = 1e9
    for j, v2 in enumerate(output2):
        if np.linalg.norm(v1 - v2) < closest:
            closest = np.linalg.norm(v1 - v2)
            closest_image = obs2[image_index][:3 * 64 * 64].reshape(3, 64, 64).transpose(1, 2, 0)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image1)
    ax[1].imshow(closest_image)
    plt.savefig('images/out{}.png'.format(i))
    plt.close(fig)

output1 = np.array(output1).reshape(len(obs1), -1)[:500]
output2 = np.array(output2).reshape(len(obs2), -1)[:500]
output = np.concatenate((output1, output2))

'''
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
outputs_2d = tsne.fit_transform(output)
'''

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
outputs_2d = pca.fit_transform(output)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
plt.scatter(outputs_2d[:NUM_OBS, 0], outputs_2d[:NUM_OBS, 1], c='r', label='Robot 1 Data')
plt.scatter(outputs_2d[NUM_OBS:, 0], outputs_2d[NUM_OBS:, 1], c='g', label='Robot 2 Data')
plt.legend()
plt.show()

plt.savefig('embeddings.png')


