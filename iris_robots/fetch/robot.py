'''/
Custom Fetch Controller
Adapted from https://github.com/rail-berkeley/robonetv2/blob/621c813cbfbca1480c7d81e73b82d446d38d1f6d/widowx_envs/widowx_envs/widowx/src/widowx_controller.py
'''

#ROBOT SPECIFIC IMPORTS
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState
from geometry_msgs.msg import *
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import moveit_commander
import actionlib

#JOINT TRANSFORMATION IMPORTS
#import modern_robotics as mr
#from modern_robotics.core import JacobianSpace, Adjoint, MatrixLog6, se3ToVec, TransInv, FKinSpace
#from pyquaternion import Quaternion
from dm_control import mjcf
from dm_robotics.moma.effectors import (arm_effector, cartesian_6d_velocity_effector)
from iris_robots.real_robot_ik.robot_ik_solver import RobotIKSolver
from iris_robots.real_robot_ik.pybullet_ik_solver import InverseKinematics


# UTILITY SPECIFIC IMPORTS
from iris_robots.transformations import euler_to_quat, quat_to_euler
from terminal_utils import run_terminal_command
from threading import Lock
import numpy as np
import rospy
import time
import os


class FetchRobot:
    def __init__(self, control_hz=20):
        rospy.init_node("fetch_controller")

        #Initialize torso
        self.torso_group = MoveGroupInterface("torso", "base_link")
        self.torso_group.moveToJointPosition(["torso_lift_joint"], [0.37])
    
        #Define move group and joint names
        self.move_group = MoveGroupInterface("arm", "base_link")
        self.group_commander = moveit_commander.MoveGroupCommander("arm")
        self.torso_joint = "torso_lift_joint"
        self.gripper_frame = 'wrist_roll_link'
        self.joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint", "upperarm_roll_joint",
            "elbow_flex_joint", "forearm_roll_joint",
            "wrist_flex_joint", "wrist_roll_joint"]


        #Set up joint listener
        self._joint_lock = Lock()
        self._angles, self._velocities = {}, {}
        rospy.Subscriber(f"/joint_states", JointState, self._joint_callback)

        #Set up gripper
        from iris_robots.fetch.gripper_controller import GripperController
        self._gripper = GripperController()

        #Set up IK Solver
        self.urdf = "../real_robot_ik/fetch/fetch_description/robots/fetch.urdf"
        self._ik = InverseKinematics(self.urdf, self.joint_names)
    


    def _joint_callback(self, msg):
        with self._joint_lock:
            for name, position, velocity in zip(msg.name, msg.position, msg.velocity):
                self._angles[name] = position
                self._velocities[name] = velocity

    def launch_robot(self):
        self._robot_process = run_terminal_command('bash' + os.getcwd() + '/fetch/launch_robot.sh')

    def kill_robot(self):
        self._robot_process.terminate()

    def update_pose(self, pos, angle, new_quat=None, duration=1.5):
        if new_quat is None:
            new_quat = euler_to_quat(angle)
        best_joints, best_pos, best_quat = self._ik.calculate_ik(pos, new_quat, self.get_joint_positions()) 
        self.update_joints(best_joints)
        return best_pos, best_quat
    
    def update_joints(self, joints):
        self.move_group.moveToJointPosition(self.joint_names, joints, wait=False)

    def update_gripper(self, close_percentage):
        self._gripper.set_position(target)

    def get_joint_positions(self):
        with self._joint_lock:
            try:
                return np.array([self._angles[k] for k in self.joint_names])
            except KeyError:
                return None

    def get_joint_velocities(self):
        with self._joint_lock:
            try:
                return np.array([self._velocities[k] for k in self.joint_names])
            except KeyError:
                return None

    def get_gripper_state(self):
        return self._gripper.get_continuous_position()

    def get_ee_pose(self):
        '''
        pose = self.group_commander.get_current_pose().pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = pose.orientation
        or_x = orientation.x
        or_y = orientation.y
        or_z = orientation.z
        or_w = orientation.w
        quat = np.array([or_x, or_y, or_z, or_w])
        '''
        joints = self.get_joint_positions()
        return self._ik.get_cartesian_pose(joints)
        return position, quat

    def get_ee_pos(self):
        return self.get_ee_pose()[0]

    def get_ee_angle(self):
        return quat_to_euler(self.get_ee_pose()[1])

if __name__ == '__main__':
    robot = FetchRobot()
    import pdb; pdb.set_trace()

