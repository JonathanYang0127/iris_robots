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


ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/purple_marker_grasp_new/combined_trajectories.npy'
ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/purple_marker_grasp_franka/combined_trajectories.npy'

with open(ROBOT_PATH_ONE, 'rb') as f:
    traj = np.load(f, allow_pickle=True)


env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
#env = RobotEnv('172.16.0.21', use_robot_cameras=True)
obs = env.reset()

class DeltaPoseToCommand:
    def __init__(self, init_obs, normalize=False, model_type='linear'):
        self._previous_obs = init_obs
        self._obs = init_obs
        self.model_type = model_type

        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import pickle

        if self.model_type == 'linear':
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/linear_cdp_model.pkl'
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        elif self.model_type == 'nonlinear':
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_adp_cdp_model_'
            if normalize:
                self.model_path += 'normalized.pt'
            else:
                self.model_path += 'unnormalized.pt'
            with open(self.model_path, 'rb') as f:
                self.model = torch.load(f)

        self.normalization_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean.pkl'
        with open(self.normalization_path, 'rb') as f:
            self.x_mean, self.x_std, self.y_mean, self.y_std = pickle.load(f)
        
        if not normalize:
            self.x_mean, self.x_std = np.zeros(self.x_mean.shape[0]), np.ones(self.x_std.shape[0])
            self.y_mean, self.y_std = np.zeros(self.y_mean.shape[0]), np.ones(self.y_std.shape[0])


    def postprocess_obs_action(self, obs, action):
        self._previous_obs = self._obs
        self._obs = obs
        adp = action.tolist()[:-1]
        adp += self._obs['current_pose'].tolist()[:-1]
        adp += self._obs['desired_pose'].tolist()[:-1]
        adp += self._previous_obs['current_pose'].tolist()[:-1]
        adp += self._previous_obs['desired_pose'].tolist()[:-1]
        adp = np.array(adp).reshape(1, -1)
        
        adp = (adp - self.x_mean) / self.x_std
        if self.model_type == 'linear': 
            return self.model.predict(adp)[0]*self.y_std + self.y_mean
        elif self.model_type == 'nonlinear':
            adp = torch.Tensor(adp).cuda()
            return self.model(adp).detach().cpu().numpy()[0]*self.y_std + self.y_mean

relabeller = DeltaPoseToCommand(obs, normalize=False, model_type='nonlinear')

index = 10
for j in range(len(traj[index]['actions'])):
    obs = env.get_observation()
    adp = pose_diff(traj[index]['observations'][j + 1]['current_pose'], traj[index]['observations'][j]['current_pose'])
    cdp = pose_diff(traj[index]['observations'][j + 1]['desired_pose'], traj[index]['observations'][j]['current_pose'])
    #action = traj[index]['actions'][j]
    #action[3:6] *= -1
    #action /= 2.5
    #action *= 20
    action = relabeller.postprocess_obs_action(obs, adp)
    action = np.concatenate((action, [0]))
    print(cdp, action)
    #action = np.clip(action, -1, 1)
    #print(obs['current_pose'], traj[index]['observations'][j]['current_pose'])
    #env.step(action)
    env.step_direct(cdp)

