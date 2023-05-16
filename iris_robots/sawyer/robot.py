'''/
Custom Fetch Controller
Adapted from https://github.com/rail-berkeley/robonetv2/blob/621c813cbfbca1480c7d81e73b82d446d38d1f6d/widowx_envs/widowx_envs/widowx/src/widowx_controller.py
'''

#ROBOT SPECIFIC IMPORTS
from sensor_msgs.msg import JointState
from geometry_msgs.msg import *
import intera_interface

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
from pathlib import Path
import numpy as np
import rospy
import time
import os


class SawyerRobot:
    def __init__(self, control_hz=20, blocking=False):
        rospy.init_node("sawyer_controller")

    
        #Set up joint control and joint names
        self.limb = intera_interface.Limb('right')
        self.joint_names = self.limb.joint_names()
        self.limb.set_joint_position_speed(0.1)

        #Set up joint listener
        self._joint_lock = Lock()
        self._angles, self._velocities = {}, {}
        rospy.Subscriber(f"/robot/joint_states", JointState, self._joint_callback)

        #Set up gripper
        self.gripper = intera_interface.Gripper('right_gripper')

        #Set up IK Solver
        current_file = Path(__file__).parent.absolute()
        self.urdf = "../real_robot_ik/sawyer/sawyer_description/urdf/sawyer_base.urdf"
        self._ik = InverseKinematics(os.path.join(current_file, self.urdf), self.joint_names)


    def _joint_callback(self, msg):
        with self._joint_lock:
            for name, position, velocity in zip(msg.name, msg.position, msg.velocity):
                self._angles[name] = position
                self._velocities[name] = velocity

    def launch_robot(self):
        self._robot_process = run_terminal_command('bash' + os.getcwd() + '/sawyer/launch_robot.sh')

    def kill_robot(self):
        pass

    def update_pose(self, pos, angle, new_quat=None, duration=1.5):
        if new_quat is None:
            new_quat = euler_to_quat(angle)
        print("Moving to", pos)
        print("Starting from", self.get_ee_pose())
        print("Joint positions", self.get_joint_positions())
        best_joints, best_pos, best_quat = self._ik.calculate_ik(pos, new_quat, self.get_joint_positions()) 
        print("Best Joints", best_joints)
        print("Best pose", best_pos, best_quat)
        print("New pose", self._ik.get_cartesian_pose(best_joints))
        self.update_joints(best_joints)
        
        return best_pos, best_quat
    
    def update_joints(self, joints):
        joint_command = {joint: value for joint, value in zip(self.joint_names, joints)}
        self.limb.move_to_joint_positions(joint_command)

    def update_gripper(self, close_percentage):
        target = 1 if close_percentage > 0.5 else 0
        if target == 1: 
            self.gripper.close()
        else:
            self.gripper.open()

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
        angle = self._angles['right_gripper_l_finger_joint']
        velocity = self._velocities['right_gripper_l_finger_joint']
        return [(0.0405 - angle) / 0.041, -velocity / 0.041]

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
        #pose = self.limb.endpoint_pose()
        #position = np.array(pose['position'])
        #quat = np.array(pose['orientation'])
        position, quat = self._ik.get_cartesian_pose(self.get_joint_positions())
        return position, quat

    def get_ee_pos(self):
        return self.get_ee_pose()[0]

    def get_ee_angle(self):
        return quat_to_euler(self.get_ee_pose()[1])

if __name__ == '__main__':
    robot = SawyerRobot()
    #robot.update_joints([0.07412109375, -0.6361875, -0.28923046875, 2.014697265625, 0.6897802734375, 0.331681640625, 0.7]) 
    robot.update_joints([ 0.63825391,  0.36787891, -1.14962207,  1.8535752 ,  1.99177148, 1.28636816,  0.13])
    import pdb; pdb.set_trace()
    pos = robot.get_ee_pos()
    angle = robot.get_ee_angle()
    pos[0] += 0.1
    robot.update_pose(pos, angle)

