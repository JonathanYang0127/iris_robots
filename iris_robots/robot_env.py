'''
Basic Robot Environment Wrapper
Robot Specific Functions: self._update_pose(), self.get_ee_pos(), self.get_ee_angle()
Camera Specific Functions: self.render_obs()
Experiment Specific Functions: self.get_info(), self.get_reward(), self.get_observation()
'''
import numpy as np
import time
import gym
from params import ROBOT_PARAMS

from iris_robots.transformations import add_angles, angle_diff
from iris_robots.camera_utils.multi_camera_wrapper import MultiCameraWrapper
from iris_robots.server.robot_interface import RobotInterface

class RobotEnv(gym.Env):

    def __init__(self, ip_address=None, robot_model='franka', control_hz=20, use_local_cameras=False, use_robot_cameras=False,
            camera_types=['cv2'], blocking=True, reverse_image=False):

        # Initialize Gym Environment
        super().__init__()

        # Physics
        self.use_desired_pose = False
<<<<<<< HEAD
        self.max_lin_vel = 2.0 #1.0
        self.max_rot_vel = 2.0
=======
>>>>>>> 0d515c16c0800a7b02ae6941c59a4f1ba27cedc9
        self.DoF = 6
        self.hz = control_hz
        self.blocking=blocking
        self.reverse_image = reverse_image

        # Robot Configuration
        self.robot_model = robot_model
        if ip_address is None:
            if robot_model == 'franka':
                from iris_robots.franka.robot import FrankaRobot
<<<<<<< HEAD
                self._robot = FrankaRobot(control_hz=self.hz, blocking=blocking)
=======
                self._robot = FrankaRobot(control_hz=self.hz)
                self.max_lin_vel = 1.0
                self.max_rot_vel = 2.0
>>>>>>> 0d515c16c0800a7b02ae6941c59a4f1ba27cedc9
            elif robot_model == 'wx200':
                from iris_robots.widowx.robot import WidowX200Robot
                self._robot = WidowX200Robot(control_hz=self.hz)
                self.max_lin_vel = 1.5
                self.max_rot_vel = 6.0
            elif robot_model == 'wx250s':
                from iris_robots.widowx.robot import WidowX250SRobot
                self._robot = WidowX250SRobot(control_hz=self.hz, blocking=blocking)
                self.max_lin_vel = 1.5
                self.max_rot_vel = 6.0
            else:
                raise NotImplementedError

        else:
            self._robot = RobotInterface(ip_address=ip_address)
            self.max_lin_vel = 1.0
            self.max_rot_vel = 2.0

        # Reset joints
        self.reset_joints = ROBOT_PARAMS[robot_model]['reset_joints']

        # Create Cameras
        self._use_local_cameras = use_local_cameras
        self._use_robot_cameras = use_robot_cameras

        if self._use_local_cameras:
            self._camera_reader = MultiCameraWrapper(camera_types=camera_types,
                    reverse=self.reverse_image)
        
        self.reset()

        if self.num_cameras == 0:
            print("Warning: No cameras found!") 

    def step(self, action):
        start_time = time.time()

        # Process Action
        assert len(action) == (self.DoF + 1)
        assert (action.max() <= 1) and (action.min() >= -1)

        pos_action, angle_action, gripper = self._format_action(action)
        lin_vel, rot_vel = self._limit_velocity(pos_action, angle_action)
        desired_pos = self._curr_pos + lin_vel
        desired_angle = add_angles(rot_vel, self._curr_angle)

        self._update_robot(desired_pos, desired_angle, gripper)

        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)
        time.sleep(sleep_left)


    def step_direct(self, action):
        start_time = time.time()
        assert len(action) == 7 #position, angle, gripper
        desired_pos = self._curr_pos + action[:3]
        desired_angle = add_angles(action[3:6], self._curr_angle)

        gripper_action = action[6]
        #gripper_action += 1
        self._update_robot(desired_pos, desired_angle, gripper_action)
        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)
        time.sleep(sleep_left)

    def reset(self):
        self._robot.update_gripper(0)
        self._robot.update_joints(self.reset_joints)
        self._desired_pose = {'position': self._robot.get_ee_pos(),
                              'angle': self._robot.get_ee_angle(),
                              'gripper': 0}
        self._default_angle = self._desired_pose['angle']
        time.sleep(2.5)
        return self.get_observation()

    def _format_action(self, action):
        '''Returns [x,y,z], [yaw, pitch, roll], close_gripper'''
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 3:
            delta_pos, delta_angle, gripper = action[:-1], default_delta_angle, action[-1]
        # elif self.DoF == 4:
        #     delta_pos, delta_angle, gripper = action[:3], action[3], action[-1]
        #     delta_angle = delta_angle.extend([0,0])
        # elif self.DoF == 5:
        #     delta_pos, delta_angle, gripper = action[:3], action[3:5], action[-1]
        #     delta_angle = delta_angle.append(0)
        elif self.DoF == 6:
            delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _limit_velocity(self, lin_vel, rot_vel):
        """Scales down the linear and angular magnitudes of the action"""

        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)

        if lin_vel_norm > 1: lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1: rot_vel = rot_vel / rot_vel_norm

        lin_vel = lin_vel * self.max_lin_vel / self.hz
        rot_vel = rot_vel * self.max_rot_vel / self.hz

        return lin_vel, rot_vel

    def _update_robot(self, pos, angle, gripper):
        feasible_pos, feasible_angle = self._robot.update_pose(pos, angle)
        self._robot.update_gripper(gripper)
        self._desired_pose = {'position': feasible_pos,
                              'angle': feasible_angle,
                              'gripper': gripper}

    @property
    def _curr_pos(self):
        if self.use_desired_pose: return self._desired_pose['position'].copy()
        return self._robot.get_ee_pos()

    @property
    def _curr_angle(self):
        if self.use_desired_pose: return self._desired_pose['angle'].copy()
        return self._robot.get_ee_angle()

    def get_images(self):
        camera_feed = []
        if self._use_local_cameras:
            camera_feed.extend(self._camera_reader.read_cameras())
        if self._use_robot_cameras:
            camera_feed.extend(self._robot.read_cameras())
        return camera_feed

    def get_state(self):
        state_dict = {}
        gripper_state = self._robot.get_gripper_state()

        state_dict['control_key'] = 'desired_pose' if \
            self.use_desired_pose else 'current_pose'

        state_dict['desired_pose'] = np.concatenate(
            [self._desired_pose['position'],
            self._desired_pose['angle'],
            [self._desired_pose['gripper']]])

        state_dict['current_pose'] = np.concatenate(
            [self._robot.get_ee_pos(),
            self._robot.get_ee_angle(),
            [gripper_state[0]]])

        state_dict['joint_positions'] = self._robot.get_joint_positions()
        state_dict['joint_velocities'] = self._robot.get_joint_velocities()
        state_dict['gripper_velocity'] = gripper_state[1]

        return state_dict

    def get_observation(self, include_images=True, include_robot_state=True):
        obs_dict = {}
        if include_images:
            obs_dict['images'] = self.get_images()
        if include_robot_state:
            state_dict = self.get_state()
            obs_dict.update(state_dict)
        return obs_dict

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self.reset_joints)
        return joint_dist < epsilon

    @property
    def num_cameras(self):
        return len(self.get_images())
