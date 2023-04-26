from iris_robots.robot_env import RobotEnv
from iris_robots.controllers.oculus_controller import VRPolicy
from iris_robots.data_collection.data_collector import DataCollector
from iris_robots.user_interface.gui import RobotGUI
import iris_robots
import numpy as np
import torch
import time
import os

policy = None

# Make the robot env
#env = RobotEnv('127.0.0.21', use_local_cameras=True) 
env = RobotEnv(robot_model='franka', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)

import pdb; pdb.set_trace()


