from iris_robots.robot_env import RobotEnv
from iris_robots.controllers.oculus_controller import VRPolicy
from iris_robots.data_collection.data_collector import DataCollector
from iris_robots.user_interface.gui import RobotGUI
from iris_robots.transformations import add_angles, angle_diff, pose_diff
import iris_robots
import numpy as np
import torch
import time
import os
from datetime import datetime
import pickle

policy = None
SAVE_DIR = 'proprioceptive_data_tc'


DATA_PATHS = [
        '/iris/u/jyang27/training_data/purple_marker_grasp_new/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/purple_marker_grasp_franka/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_blue_floral/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_mixed_floral/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_gray/combined_trajectories.npy',
]
traj = []

for path in DATA_PATHS:
    traj.extend(np.load(path, allow_pickle=True))

# Make the robot env
env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)

for i in range(200):
    env.reset()
    trajectory = dict()
    trajectory['observations'] = []
    trajectory['actions'] = []
    index = np.random.randint(len(traj))
    for j in range(len(traj[index]['actions']) - 1):
        obs = env.get_observation()
        next_desired_pose = traj[index]['observations'][j+1]['desired_pose']
        #next_desired_pose += np.random.normal(scale=0.05, size=(7,))
        #cdp = pose_diff(next_desired_pose, obs['current_pose'])
        cdp = pose_diff(next_desired_pose, traj[index]['observations'][j]['current_pose'])
        env.step_direct(cdp)
        trajectory['observations'].append(obs)
        trajectory['actions'].append(traj[index]['actions'][j])
    
    now = datetime.now() 
    time_str = now.strftime('%m_%d_%y-%H_%M_%S.pkl')
    save_path = os.path.join(SAVE_DIR, time_str)
    with open(save_path, 'wb+') as f:
        pickle.dump(trajectory, f)

