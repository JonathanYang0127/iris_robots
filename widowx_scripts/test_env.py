from iris_robots.robot_env import RobotEnv
import numpy as np
import torch
import time
import os


# Make the robot env
env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=True)
env.reset()

import pdb; pdb.set_trace()

for i in range(10):
    env.step(np.array([0.5, 0, 0, 0]))
    print(env.get_observation())
    #time.sleep(1.0)
