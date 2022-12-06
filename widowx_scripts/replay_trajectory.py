from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from iris_robots.robot_env import RobotEnv

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
from datetime import datetime

import rlkit.torch.pytorch_util as ptu


ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/purple_marker_grasp_new/combined_trajectories.npy'
ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/purple_marker_grasp_franka/combined_trajectories.npy'

with open(ROBOT_PATH_TWO, 'rb') as f:
    traj = np.load(f, allow_pickle=True)



env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
#env = RobotEnv('172.16.0.21', use_robot_cameras=True)
env.reset()

index = 0
for j in range(len(traj[0]['actions'])):
    obs = env.get_observation()
    action = traj[index]['actions'][j]
    action[3:6] *= -1
    env.step(action)

