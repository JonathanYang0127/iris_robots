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
env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
env.reset()
import time; time.sleep(5)
controller = VRPolicy(pos_action_gain=[20, 20, 20],
                      rot_action_gain=20, rmat_reorder=[2, 1, -3, 4])

# Make the data collector
log_dir = os.path.join(os.path.dirname(iris_robots.__file__), 'training_data')
log_dir = os.path.join(log_dir, "wx250_purple_marker_grasp_blue_nodesired_3dof")
data_collector = DataCollector(env=env, controller=controller, policy=policy, log_dir=log_dir)

# Collect and save trajectories
for i in range(50):
    print("Trajectory {}".format(i))
    data_collector.collect_trajectory()
    time.sleep(5)




