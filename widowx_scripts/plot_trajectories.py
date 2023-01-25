import numpy as np
import torch
import pickle
import argparse

import rlkit.torch.pytorch_util as ptu
from iris_robots.transformations import add_angles, angle_diff

#ROBOT_PATH = '/iris/u/jyang27/training_data/purple_marker_grasp_new/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/proprioceptive_data_collection/proprioceptive_data_tc_nodesired/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/dev/iris_robots/iris_robots/training_data/wx250_purple_marker_grasp_blue_nodesired/combined_trajectories.npy'
ROBOT_PATH = '/iris/u/jyang27/training_data/franka_black_marker_grasp_blue_nodesired/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_black_marker_grasp_blue_nodesired_control2/combined_trajectories.npy'

with open(ROBOT_PATH, 'rb') as f:
    trajectories = np.load(f, allow_pickle=True)

def process_image(image, downsample=False):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    if downsample:
        image = image[:,::2, ::2]
    return image.flatten()


actions = []
commanded_delta_pose = []        #next desired pose - current pose
achieved_delta_pose = []         #next achieved pose - current pose
delta_joints = []                #delta joints
current_poses = []

def pose_diff(target, source, sin_angle=False):
    diff = np.zeros(len(target))
    diff[:3] = target[:3] - source[:3]
    diff[3:6] = angle_diff(target[3:6], source[3:6])
    diff[6] = target[6] - source[6]
    return diff


def limit_velocity(action):
    """Scales down the linear and angular magnitudes of the action"""
    max_lin_vel = 1.5
    max_rot_vel = 8.0
    hz = 20

    lin_vel = action[:3]
    rot_vel = action[3:6]
    
    lin_vel_norm = np.linalg.norm(lin_vel)
    rot_vel_norm = np.linalg.norm(rot_vel)

    if lin_vel_norm > 1: lin_vel = lin_vel / lin_vel_norm
    if rot_vel_norm > 1: rot_vel = rot_vel / rot_vel_norm

    lin_vel = lin_vel * max_lin_vel / hz
    rot_vel = rot_vel * max_rot_vel / hz

    return np.concatenate((lin_vel, rot_vel))

path = trajectories[20]
for t in range(0, len(path['observations']) // 2):
    action = limit_velocity(path['actions'][t]).tolist()
    current_pose = path['observations'][t]['current_pose'].tolist()
    desired_pose = path['observations'][t]['desired_pose']
    joints = path['observations'][t]['joint_positions']
    next_achieved_pose = path['observations'][t + 1]['current_pose']
    next_desired_pose = path['observations'][t + 1]['desired_pose']
    next_joints = path['observations'][t + 1]['joint_positions']

    adp = pose_diff(next_achieved_pose,  current_pose).tolist()[:-1]
    delta_joint = (next_joints - joints).tolist()
    #adp = next_desired_pose[:-1]  

    ##import pdb; pdb.set_trace()
    history = []
    for i in range(0, 2):
        index = 0 if t-i < 0 else t-i
        history += path['observations'][index]['current_pose'].tolist()[:-1]
        #history += path['observations'][index]['desired_pose'].tolist()[:-1]
        history += path['observations'][index]['joint_positions'].tolist()
    #adp += path['observations'][t + 1]['current_pose'].tolist()
        
    #adp += (path['observations'][t]['current_pose']).tolist()
    #adp += (path['observations'][t - 1]['current_pose']).tolist() 
    cdp = pose_diff(next_desired_pose, current_pose).tolist()[:-1]
    adp += history
        
    actions.append(action)
    achieved_delta_pose.append(adp) 
    commanded_delta_pose.append(cdp)
    delta_joints.append(delta_joint)
    current_poses.append(current_pose)

current_poses = np.array(current_poses)
achieved_delta_pose = np.array(achieved_delta_pose)
commanded_delta_pose = np.array(commanded_delta_pose)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

X = current_poses[:, 0]
Y = current_poses[:, 1]
Z = current_poses[:, 2]
U = achieved_delta_pose[:, 0]
V = achieved_delta_pose[:, 1]
W = achieved_delta_pose[:, 2]

U2 = commanded_delta_pose[:, 0]
V2 = commanded_delta_pose[:, 1]
W2 = commanded_delta_pose[:, 2]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W)
#ax.scatter(X, Y, Z)
ax.set_xlim([0, 0.3])
ax.set_ylim([0, 0.3])
ax.set_zlim([0, 0.2])
plt.savefig('plot2d.png')
ax.quiver(X[::10], Y[::10], Z[::10], U2[::10], V2[::10], W2[::10], color='red')
plt.savefig('plot2d_cdp.png')


import torch
import torch.nn as nn
model = nn.Sequential(
        nn.Linear(achieved_delta_pose.shape[1], 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 3)
        ).cuda()


#model = InverseDynamicsModel(x_train.shape[1], 7).cuda()
#model.load_state_dict(torch.load('new2/checkpoints_cdp_normalized_bigger/output_100000.pt'))
model = torch.load('/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_adp_cdp_xyz_model_unnormalized.pt')
x_test_torch = torch.from_numpy(achieved_delta_pose).cuda().float()
model_cdp = model(x_test_torch).detach().cpu().numpy()

U3 = model_cdp[:, 0]
V3 = model_cdp[:, 1]
W3 = model_cdp[:, 2]

#plt.clf()
ax.quiver(X[::10], Y[::10], Z[::10], U3[::10], V3[::10], W3[::10], color='green')
#plt.show()
plt.savefig('plot2d_cdp_model.png')
 


