from iris_robots.robot_env import RobotEnv
import numpy as np
import torch
import time
import os


# Make the robot env
env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
env.reset()

import pdb; pdb.set_trace()


