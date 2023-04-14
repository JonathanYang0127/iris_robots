import numpy as np
import pybullet as p
import rospy
from threading import Lock
import time


class InverseKinematics():
    def __init__(self, urdf, joint_names, joints_min=None, joints_max=None):
        #IK position is assumed relative to last joint name

        p.connect(p.DIRECT)
        
        self._robot_id = p.loadURDF(urdf)
        self.num_urdf_joints = p.getNumJoints(self._robot_id)
        self.joint_names = joint_names
        self.joint_map = [0] * len(joint_names)
        self.num_joints = len(joint_names)

        #Figure out which joints in the URDF correspond to the joint names
        for i in range(self.num_urdf_joints):
            info = p.getJointInfo(self._robot_id, i)
            for j, name in enumerate(joint_names):
                if info[1].decode('UTF-8') == name:
                    self.joint_map[j] = i
        self.ik_link_id = self.joint_map[-1]
        

        #Create min and max joint limits
        self.joint_limits_min = [-3.8] * self.num_urdf_joints
        self.joint_limits_max = [3.8] * self.num_urdf_joints
        self.joint_limits_min[2] = 0.37
        self.joint_limits_max[2] = 0.37
        if joints_min is not None:
            for i in range(len(joints_min)):
                self.joint_limits_min[self.joint_map[i]] = joints_max[i]
        if joints_max is not None:
            for i in range(len(joints_max)):
                self.joint_limits_max[self.joint_map[i]] = joints_max[i]
        p.resetBasePositionAndOrientation(self._robot_id, [0, 0, 0], [0, 0, 0, 1])


    def _reset_pybullet(self, joint_angles = None):
        '''
        Reset pybullet sim to current joint angles
        '''
        assert joint_angles is None or len(joint_angles) == self.num_joints
        if joint_angles is None:
            joint_angles = self.get_joint_angles()
        for i, angle in enumerate(joint_angles):
            p.resetJointState(self._robot_id, self.joint_map[i], angle)


    def get_cartesian_pose(self, joint_angles):
        '''
        Get xyz pose for arm (computes from simulation)
        '''
        self._reset_pybullet(joint_angles)
        position, quat = p.getLinkState(self._robot_id, self.ik_link_id, computeForwardKinematics=1)[4:6]
        return np.array(list(position), dtype='float32'), np.array(list(quat), dtype='float32')


    def calculate_ik(self, targetPos, targetQuat, joint_angles, threshold=1e-5, maxIter=1000):
        '''
        Compute ik solution given pose
        '''
        closeEnough = False
        iter_count = 0
        dist2 = None

        best_joints, best_pos, best_quat, best_dist = None, None, None, float('inf')
        self._reset_pybullet(joint_angles) 
        
        while (not closeEnough and iter_count < maxIter):
            jointPoses = list(p.calculateInverseKinematics(self._robot_id, self.ik_link_id, targetPos, targetQuat, self.joint_limits_min, self.joint_limits_max))
            for i in range(self.num_joints):
                jointPoses[i] = max(min(jointPoses[i], self.joint_limits_max[i]), self.joint_limits_min[i])
                p.resetJointState(self._robot_id, i, jointPoses[i])

            ls = p.getLinkState(self._robot_id, self.ik_link_id, computeForwardKinematics=1)
            newPos, newQuat = ls[4], ls[5]
            dist2 = sum([(targetPos[i] - newPos[i]) ** 2 for i in range(3)])
            closeEnough = dist2 < threshold
            iter_count += 1

            if dist2 < best_dist:
                best_joints, best_pos, best_quat, best_dist = jointPoses, newPos, newQuat, dist2

        #Filter for useful joints
        self.ik_offset = 5
        best_useful_joints = [0] * self.num_joints
        for i in range(self.num_joints):
            best_useful_joints[i] = best_joints[self.ik_offset + i]
        return best_useful_joints, best_pos, best_quat




if __name__ == '__main__':
    rospy.init_node('IK_Node')
    urdf = "fetch/fetch_description/robots/fetch.urdf"
    joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint", "upperarm_roll_joint",
            "elbow_flex_joint", "forearm_roll_joint",
            "wrist_flex_joint", "wrist_roll_joint"]
    k = InverseKinematics(urdf, joint_names)

    print(k.get_cartesian_pose([0, -1.0, 0, 1.5, 0,  1.2, 0]))
    print(k.get_cartesian_pose([0, -0.5, 0, 1.5, 0,  1.2, 0]))
    print("HI")
