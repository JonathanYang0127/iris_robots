from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from iris_robots.transformations import add_angles, angle_diff, pose_diff
from iris_robots.robot_env import RobotEnv

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import torch

import rlkit.torch.pytorch_util as ptu


#ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_blue_nodesired_control3/combined_trajectories.npy'
ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_nodesired_control3_ee2/wx250_black_marker_grasp_gray_nodesired_control3_ee2/combined_trajectories.npy'

with open(ROBOT_PATH, 'rb') as f:
    traj = np.load(f, allow_pickle=True)


env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=True)
#env = RobotEnv('172.16.0.21', use_robot_cameras=True)
obs = env.reset()

index = 20
for j in range(0, len(traj[index]['actions']), 20):
    obs = env.get_observation()
    adp = pose_diff(traj[index]['observations'][j + 20]['current_pose'], traj[index]['observations'][j]['current_pose'])
    cdp = pose_diff(traj[index]['observations'][j + 20]['desired_pose'], traj[index]['observations'][j]['current_pose'])
    cdp[6] = env._robot._gripper.normalize(cdp[6])
    env.step_direct(adp)

