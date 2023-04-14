import numpy as np
from dm_control import mjcf
from dm_robotics.moma.effectors import (arm_effector,
										cartesian_6d_velocity_effector)
from scipy.spatial.transform import Rotation as R
from iris_robots.real_robot_ik.arm import FrankaArm, WidowX200Arm


arm = FrankaArm()
physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
import pdb; pdb.set_trace()

