from iris_robots.transformations import add_angles, angle_diff, pose_diff
from iris_robots.robot_env import RobotEnv
from PIL import Image

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import torch

import time
#import rlkit.torch.pytorch_util as ptu


#ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_blue_nodesired_control3/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_gray_nodesired_control3_ee2/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_shelf_close_camera2/wx250_shelf_close_grasp_camera2/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_shelf_close_camera2/wx250_shelf_close_grasp_reverse_camera2/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/dev/iris_robots/iris_robots/training_data/wx250_shelf_close_bottom/combined_trajectories.npy'
#ROBOT_PATH = '/iris/u/jyang27/dev/iris_robots/iris_robots/training_data/wx250_pickplace/combined_trajectories.npy'
#ROBOT_PATH = '/home/locobot/dev_jonathan/iris_robots/iris_robots/training_data/sawyer_shelf/sawyer_shelf_forward/Wed_May_17_15:28:50_2023.npy'
#ROBOT_PATH = '/home/locobot/dev_jonathan/iris_robots/iris_robots/training_data/sawyer_shelf/sawyer_shelf_reverse/Wed_May_17_16:01:23_2023.npy'
#ROBOT_PATH = '/home/locobot/dev_jonathan/iris_robots/iris_robots/training_data/sawyer_shelf/sawyer_shelf_bottom/Wed_May_17_16:27:00_2023.npy'
ROBOT_PATH = '/home/locobot/dev_jonathan/iris_robots/iris_robots/training_data/sawyer_black_marker_pickplace/sawyer_black_marker_pickplace_mixed_floral/Wed_May_17_14:19:49_2023.npy'

with open(ROBOT_PATH, 'rb') as f:
    traj = np.load(f, allow_pickle=True).item()


#env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
#env = RobotEnv('172.16.0.21', use_robot_cameras=True)
env = RobotEnv(robot_model='sawyer', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False, reverse_image=True)

obs = env.reset()

class DeltaPoseToCommand:
    def __init__(self, init_obs, robot_type, normalize=False, model_type='nonlinear'):
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
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_{}_adp_cdp_xyz_model_'.format(robot_type)
            self.angle_model_path =  '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_{}_adp_cdp_angle_model_'.format(robot_type)
            if normalize:
                self.model_path += 'normalized.pt'
                self.angle_model_path += 'normalized.pt'
            else:
                self.model_path += 'unnormalized.pt'
                self.angle_model_path += 'unnormalized.pt'
            with open(self.model_path, 'rb') as f:
                self.model = torch.load(f)
            with open(self.angle_model_path, 'rb') as f:
                self.angle_model = torch.load(f)

        self.normalization_path_xyz = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean_adp_cdp_xyz.pkl'
        with open(self.normalization_path_xyz, 'rb') as f:
            self.x_mean_xyz, self.x_std_xyz, self.y_mean_xyz, self.y_std_xyz = pickle.load(f)

        self.normalization_path_angle = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean_adp_cdp_angle.pkl'
        with open(self.normalization_path_angle, 'rb') as f:
            self.x_mean_angle, self.x_std_angle, self.y_mean_angle, self.y_std_angle = pickle.load(f)

        if not normalize:
            self.x_mean_xyz, self.x_std_xyz = np.zeros(self.x_mean_xyz.shape[0]), np.ones(self.x_std_xyz.shape[0])
            self.y_mean_xyz, self.y_std_xyz = np.zeros(self.y_mean_xyz.shape[0]), np.ones(self.y_std_xyz.shape[0])

            self.x_mean_angle, self.x_std_angle = np.zeros(self.x_mean_angle.shape[0]), np.ones(self.x_std_angle.shape[0])
            self.y_mean_angle, self.y_std_angle = np.zeros(self.y_mean_angle.shape[0]), np.ones(self.y_std_angle.shape[0])

    def set_init_obs(self, obs):
        self._previous_obs = obs
        self._obs = obs

    def postprocess_obs_action(self, obs, action):
        self._previous_obs = self._obs
        self._obs = obs
        adp = action.tolist()[:-1]
        adp += self._obs['current_pose'].tolist()[:-1]
        adp += self._obs['joint_positions'].tolist()
        #adp += self._obs['desired_pose'].tolist()[:-1]
        adp += self._previous_obs['current_pose'].tolist()[:-1]
        adp += self._previous_obs['joint_positions'].tolist()
        #adp += self._previous_obs['desired_pose'].tolist()[:-1]
        adp = np.array(adp).reshape(1, -1)

        adp = (adp - self.x_mean_xyz) / self.x_std_xyz
        if self.model_type == 'linear':
            return self.model.predict(adp)[0]*self.y_std + self.y_mean
        elif self.model_type == 'nonlinear':
            adp = torch.Tensor(adp).cuda()
            xyz = self.model(adp).detach().cpu().numpy()[0]*self.y_std_xyz + self.y_mean_xyz
            angle_repr = self.angle_model(adp).detach().cpu().numpy()[0]
            angle = np.arctan2(angle_repr[:3], angle_repr[3:6])
            return np.concatenate((xyz, angle))


#relabeller = DeltaPoseToCommand(obs, 'wx250s', normalize=False, model_type='nonlinear')
index = 1
images = []
prev_gripper = 0
for j in range(len(traj['actions']) - 1):
    obs = env.get_observation()
    image = obs['images'][1]['array']
    images.append(Image.fromarray(image))
    print(traj['observations'][j + 1]['desired_pose'])
    #adp = pose_diff(traj[index]['observations'][j + 1]['current_pose'], obs['current_pose'])
    #adp = pose_diff(traj['observations'][j + 1]['current_pose'], traj['observations'][j]['current_pose'])
    #cdp = pose_diff(traj['observations'][j + 1]['desired_pose'][:7], traj['observations'][j]['current_pose'][:7])
    joints = traj['observations'][j+1]['joint_positions']
    env._robot.update_joints(joints)
    if prev_gripper == 0 and traj['observations'][j + 1]['desired_pose'][7] == 1:
        env._robot.update_gripper(traj['observations'][j + 1]['desired_pose'][7])
        prev_gripper = 1
    elif prev_gripper == 1 and traj['observations'][j + 1]['desired_pose'][7] == 0:
        env._robot.update_gripper(traj['observations'][j + 1]['desired_pose'][7])
        prev_gripper = 0
    #action = traj[index]['actions'][j]
    #action[3:6] *= -1
    #action /= 2.5
    #action *= 20
    #action = relabeller.postprocess_obs_action(obs, adp)
    #action = np.concatenate((action, [0]))
    #print(cdp, action)
    #action = np.clip(action, -1, 1)
    #print(obs['current_pose'], traj[index]['observations'][j]['current_pose'])
    #env.step(action)
    #cdp[6] = env._robot._gripper.normalize(cdp[6])
    #print(cdp)
    #env.step_direct(cdp)
    time.sleep(0.1)

import time
print("ASD")
images[0].save('{}/eval_{}.gif'.format('eval_videos', int(time.time())),
                            format='gif', append_images=images[1:],
                            save_all=True, duration=100, loop=0)
                                                                  
