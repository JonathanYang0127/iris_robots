'''
Custom WidowX Controller
Adapted from https://github.com/rail-berkeley/robonetv2/blob/621c813cbfbca1480c7d81e73b82d446d38d1f6d/widowx_envs/widowx_envs/widowx/src/widowx_controller.py
'''

# ROBOT SPECIFIC IMPORTS
from iris_robots.real_robot_ik.robot_ik_solver import RobotIKSolver
from iris_robots.widowx.custom_gripper_controller import GripperController
from interbotix_xs_modules.arm import InterbotixArmXSInterface, InterbotixArmXSInterface, InterbotixRobotXSCore, InterbotixGripperXSInterface
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand


#JOINT TRANSFORMATION IMPORTS
import modern_robotics as mr
from modern_robotics.core import JacobianSpace, Adjoint, MatrixLog6, se3ToVec, TransInv, FKinSpace
from pyquaternion import Quaternion

# UTILITY SPECIFIC IMPORTS
from iris_robots.transformations import euler_to_quat, quat_to_euler
from terminal_utils import run_terminal_command
from threading import Lock
import numpy as np
import torch
import rospy
import time
import os


def compute_joint_velocities_from_cartesian(Slist, M, T, thetalist_current):
    """Computes inverse kinematics in the space frame for an open chain robot
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist_current: An initial guess of joint angles that are close to
                       satisfying Tsd
    """
    thetalist = np.array(thetalist_current).copy()
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb),
                se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    theta_vel = np.dot(np.linalg.pinv(JacobianSpace(Slist,
                                                    thetalist)), Vs)
    return theta_vel


class ModifiedInterbotixArmXSInterface(InterbotixArmXSInterface):
    def __init__(self, *args, **kwargs):
        super(ModifiedInterbotixArmXSInterface, self).__init__(*args, **kwargs)
        self.waist_index = self.group_info.joint_names.index("waist")

    def set_ee_pose_matrix_fast(self, T_sd, custom_guess=None, execute=True):
        """
        this version of set_ee_pose_matrix does not set the velocity profile registers in the servos and therefore runs faster
        """
        if (custom_guess is None):
            initial_guesses = self.initial_guesses
        else:
            initial_guesses = [custom_guess]

        for guess in initial_guesses:
            theta_list, success = mr.IKinSpace(self.robot_des.Slist, self.robot_des.M, T_sd, guess, 0.001, 0.001)
            solution_found = True

            # Check to make sure a solution was found and that no joint limits were violated
            if success:
                theta_list = [int(elem * 1000) / 1000.0 for elem in theta_list]
                for x in range(self.group_info.num_joints):
                    if not (self.group_info.joint_lower_limits[x] <= theta_list[x] <=
                            self.group_info.joint_upper_limits[x]):
                        solution_found = False
                        break
            else:
                solution_found = False

            if solution_found:
                if execute:
                    self.publish_positions_fast(theta_list)
                    self.T_sb = T_sd
                return theta_list, True
            else:
                rospy.loginfo("Guess failed to converge...")

        rospy.loginfo("No valid pose could be found")
        return theta_list, False

    def publish_positions_fast(self, positions):
        self.joint_commands = list(positions)
        joint_commands = JointGroupCommand(self.group_name, self.joint_commands)
        self.core.pub_group.publish(joint_commands)


class WidowXRobot:
    def __init__(self, control_hz=20, robot_model='wx250s', blocking=True):
        self.robot_model = robot_model
        self.dxl = InterbotixRobotXSCore(robot_model, None, True)
        self.arm = ModifiedInterbotixArmXSInterface(self.dxl, robot_model, 'arm', 2.0, 0.3)

        self._joint_lock = Lock()
        self._angles, self._velocities = {}, {}
        rospy.Subscriber(f"/{robot_model}/joint_states", JointState, self._joint_callback)
        time.sleep(1)
        self._n_errors = 0

        self._upper_joint_limits = np.array(self.arm.group_info.joint_upper_limits)
        self._lower_joint_limits = np.array(self.arm.group_info.joint_lower_limits)
        self._qn = self.arm.group_info.num_joints

        #TODO: Change to allow for different "neutral" base rotations
        self.joint_names = self.arm.group_info.joint_names
        DEFAULT_ROTATION = np.array([[0, 0, 1.0],
                                     [0, 1.0, 0],
                                     [-1.0, 0, 0]])
        self.default_rot = DEFAULT_ROTATION

        self._gripper = GripperController(robot_model)
        self.blocking = blocking

        #self._ik_solver = RobotIKSolver(None, control_hz=control_hz, arm_name='wx200')

    def _joint_callback(self, msg):
        with self._joint_lock:
            for name, position, velocity in zip(msg.name, msg.position, msg.velocity):
                self._angles[name] = position
                self._velocities[name] = velocity

    def launch_robot(self):
        self._robot_process = run_terminal_command('bash ' + os.getcwd() + '/widowx/launch_robot.sh')

    def kill_robot(self):
        self._robot_process.terminate()

    def update_pose(self, pos, angle, duration=1.5):
        '''Expect [x,y,z], [yaw, pitch, roll]'''

        new_pose = np.eye(4)
        new_pose[:3, -1] = pos
        new_quat = Quaternion(euler_to_quat(angle))
        new_pose[:3, :3] = new_quat.rotation_matrix

        if not self.blocking:
            solution, success = self.arm.set_ee_pose_matrix_fast(new_pose, custom_guess=self.get_joint_positions(),
                                                                     execute=True)
        else:
            solution, success = self.arm.set_ee_pose_matrix(new_pose, custom_guess=self.get_joint_positions(),
                                                                moving_time=duration, accel_time=duration * 0.45)

        return pos, angle

    def update_joints(self, joints, duration=1.5):
        if not self.blocking:
            self.arm.publish_positions_fast(joints)
        else:
            self.arm.publish_positions(joints, moving_time=duration, accel_time=duration * 0.45)

    def update_gripper(self, close_percentage):
        desired_gripper = np.clip(1 - close_percentage, 0.05, 1)
        self._gripper.set_continuous_position(desired_gripper)
        if self.blocking:
            time.sleep(0.5)
            '''
            while np.abs(self._gripper.get_continuous_position() - desired_gripper) > 0.05:
                print(self._gripper.get_continuous_position(), desired_gripper)
                self._gripper.set_continuous_position(desired_gripper)
                if self._gripper._velocities['left_finger'] < 0.01:
                    break
                time.sleep(0.01)
            '''

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
        state_1 = self._gripper.get_continuous_position()
        time_1 = time.time()

        state_2 = self._gripper.get_continuous_position()
        time_2 = time.time()

        vel = (state_2 - state_1) / (time_2 - time_1)

        return np.array([state_2, vel])

    def get_ee_pose(self):
        joint_positions = list(self.dxl.joint_states.position[self.arm.waist_index:(self._qn + self.arm.waist_index)])
        pose = mr.FKinSpace(self.arm.robot_des.M, self.arm.robot_des.Slist, joint_positions)

        return pose[:3, -1], np.array(Quaternion(matrix=pose[:3, :3]).elements)

    def get_ee_pos(self):
        '''Returns [x,y,z]'''
        return self.get_ee_pose()[0]

    def get_ee_angle(self):
        '''Returns [yaw, pitch, roll]'''
        return quat_to_euler(self.get_ee_pose()[1])

class WidowX200Robot(WidowXRobot):
    def __init__(self, control_hz=20, blocking=True):
        super().__init__(control_hz=control_hz, robot_model='wx200', blocking=blocking)

class WidowX250SRobot(WidowXRobot):
    def __init__(self, control_hz=20, blocking=True):
        super().__init__(control_hz=control_hz, robot_model='wx250s', blocking=blocking)

if __name__ == '__main__':
    robot = WidowX250SRobot()
    import pdb; pdb.set_trace()
