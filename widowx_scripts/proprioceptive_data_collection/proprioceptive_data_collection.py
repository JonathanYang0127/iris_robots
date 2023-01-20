from iris_robots.robot_env import RobotEnv
from iris_robots.controllers.oculus_controller import VRPolicy
from iris_robots.data_collection.data_collector import DataCollector
from iris_robots.user_interface.gui import RobotGUI
import iris_robots
import numpy as np
import torch
import time
import os
from datetime import datetime
import pickle

policy = None
SAVE_DIR = 'proprioceptive_data'


# Make the robot env
env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
env.reset()

for i in range(200):
    env.reset()
    trajectory = dict()
    trajectory['observations'] = []
    trajectory['actions'] = []
    goal = np.random.uniform(low=[0.18, -0.22, 0.055], high=[0.3, 0.17, 0.19])
    try:
        for j in range(120):
            state = env.get_state()
            magnitude = np.random.uniform(-2.5, -0.5)
            magnitude = np.power(10, magnitude)
            action_xyz = goal - env.get_state()['current_pose'][:3]
            action_xyz = action_xyz / np.linalg.norm(action_xyz) * magnitude
            action_xyz += np.random.uniform(low=-0.01, high=-0.01, size=(3,))
            action_angle = np.random.uniform(low=-1.0, high=1.0, size=(3,))
            action_angle[1] = np.random.uniform(low=-0.8, high=1.0)
            action = np.concatenate((action_xyz, action_angle, [0]))
            env.step(action)
            trajectory['observations'].append(state)
            trajectory['actions'].append(action)
    
        now = datetime.now() 
        time_str = now.strftime('%m_%d_%y-%H_%M_%S.pkl')
        save_path = os.path.join(SAVE_DIR, time_str)
        with open(save_path, 'wb+') as f:
            pickle.dump(trajectory, f)
    except:
        continue

