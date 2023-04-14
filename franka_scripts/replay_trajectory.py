from iris_robots.robot_env import RobotEnv
from iris_robots.data_collection.data_collector import DataCollector
from iris_robots.user_interface.gui import RobotGUI
from iris_robots.transformations import add_angles, angle_diff, pose_diff
import iris_robots
import numpy as np
import torch
import time
import os
from datetime import datetime
import pickle

policy = None
SAVE_DIR = 'proprioceptive_data_tc_nodesired_new'

DATA_PATHS = [
    '/iris/u/jyang27/training_data/franka_black_marker_grasp_blue_nodesired/combined_trajectories.npy'
]
traj = []


for path in DATA_PATHS:
    traj.extend(np.load(path, allow_pickle=True))


class DeltaPoseToCommand:
    def __init__(self, init_obs, normalize=False, model_type='linear'):
        self._previous_obs = init_obs
        self._obs = init_obs
        self.model_type = model_type

        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error
        import pickle

        if self.model_type == 'linear':
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/linear_cdp_model.pkl'
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        elif self.model_type == 'nonlinear':
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_adp_cdp_xyz_model_'
            self.angle_model_path =  '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_adp_cdp_angle_model_'
            if normalize:
                self.model_path += 'normalized.pt'
                self.angle_model_path += 'normalized.pt'
            else:
                self.model_path += 'unnormalized.pt'
                self.angle_model_path += 'unnormalized.pt'
            with open(self.model_path, 'rb') as f:
                self.model = torch.load(f)
            with open(self.angle_model_path, 'rb') as f:
                self.angle_model = torch.load(f)

        self.normalization_path_xyz = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean_adp_cdp_xyz.pkl'
        with open(self.normalization_path_xyz, 'rb') as f:
            self.x_mean_xyz, self.x_std_xyz, self.y_mean_xyz, self.y_std_xyz = pickle.load(f)

        self.normalization_path_angle = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean_adp_cdp_angle.pkl'
        with open(self.normalization_path_angle, 'rb') as f:
            self.x_mean_angle, self.x_std_angle, self.y_mean_angle, self.y_std_angle = pickle.load(f)

        if not normalize:
            self.x_mean_xyz, self.x_std_xyz = np.zeros(self.x_mean_xyz.shape[0]), np.ones(self.x_std_xyz.shape[0])
            self.y_mean_xyz, self.y_std_xyz = np.zeros(self.y_mean_xyz.shape[0]), np.ones(self.y_std_xyz.shape[0])

            self.x_mean_angle, self.x_std_angle = np.zeros(self.x_mean_angle.shape[0]), np.ones(self.x_std_angle.shape[0])
            self.y_mean_angle, self.y_std_angle = np.zeros(self.y_mean_angle.shape[0]), np.ones(self.y_std_angle.shape[0])

            
    def postprocess_obs_action(self, obs, action):
        self._previous_obs = self._obs
        self._obs = obs
        adp = action.tolist()[:-1]
        adp += self._obs['current_pose'].tolist()[:-1]
        adp += self._obs['joint_positions'].tolist()    
        #adp += self._obs['desired_pose'].tolist()[:-1]
        adp += self._previous_obs['current_pose'].tolist()[:-1]
        adp += self._previous_obs['joint_positions'].tolist()
        #adp += self._previous_obs['desired_pose'].tolist()[:-1]
        adp = np.array(adp).reshape(1, -1)

        adp = (adp - self.x_mean_xyz) / self.x_std_xyz
        if self.model_type == 'linear':
            return self.model.predict(adp)[0]*self.y_std + self.y_mean
        elif self.model_type == 'nonlinear':
            adp = torch.Tensor(adp).cuda()
            xyz = self.model(adp).detach().cpu().numpy()[0]*self.y_std_xyz + self.y_mean_xyz
            angle_repr = self.angle_model(adp).detach().cpu().numpy()[0]
            angle = np.arctan2(angle_repr[:3], angle_repr[3:6])
            return np.concatenate((xyz, angle))


# Make the robot env
env = RobotEnv('172.16.0.21')
obs = env.reset()
#obs = traj[0]['observations'][0]
#relabeller = DeltaPoseToCommand(obs, normalize=False, model_type='nonlinear')

for i in range(200):
    env.reset()
    trajectory = dict()
    trajectory['observations'] = []
    trajectory['actions'] = []
    current_poses = []
    achieved_delta_pose = []
    commanded_delta_pose = []
    model_cdps = []

    index = np.random.randint(len(traj))
    
    #relabeller = DeltaPoseToCommand(obs, normalize=False, model_type = 'nonlinear')

    for j in range(len(traj[index]['actions']) - 1):
        obs = env.get_observation()
        action = traj[index]['actions'][j]
        
        current_pose = obs['current_pose']
        current_pose_traj = traj[index]['observations'][j]['current_pose']
        next_achieved_pose = traj[index]['observations'][j+1]['current_pose']
        next_desired_pose = traj[index]['observations'][j+1]['desired_pose']
        adp = pose_diff(next_achieved_pose, obs['current_pose'])
        cdp = pose_diff(next_desired_pose, current_pose_traj)#obs['current_pose'])

        #model_cdp = relabeller.postprocess_obs_action(obs, adp)
        #model_cdps.append(model_cdp)
        #cdp_command = np.concatenate((model_cdp, cdp[3:]))
        #cdp_command = np.concatenate((model_cdp, [env._robot._gripper.normalize(cdp[-1])]))
        #print('**********************************', cdp, cdp_command)
        #print(next_achieved_pose, traj[index]['observations'][j]['current_pose'])
        #cdp = pose_diff(next_achieved_pose, traj[index]['observations'][j]['current_pose'])
        #action = np.concatenate((cdp[:3], [0]))
        
        pos_action, angle_action, gripper = env._format_action(action)
        lin_vel, rot_vel = env._limit_velocity(pos_action, angle_action)
        cdp = np.concatenate((lin_vel, rot_vel, [action[6] - 1]))
        #env.step(action)
        env.step_direct(cdp)
        
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)

        achieved_delta_pose.append(adp)
        commanded_delta_pose.append(cdp)
        current_poses.append(current_pose)
        
    
    now = datetime.now() 
    time_str = now.strftime('%m_%d_%y-%H_%M_%S.pkl')
    save_path = os.path.join(SAVE_DIR, time_str)
    with open(save_path, 'wb+') as f:
        pickle.dump(trajectory, f)


    '''
    num_steps = len(current_poses)
    current_poses = np.array(current_poses)[:num_steps // 2]
    achieved_delta_pose = np.array(achieved_delta_pose)[:num_steps // 2]
    commanded_delta_pose = np.array(commanded_delta_pose)[:num_steps//2]
    model_cdps = np.array(model_cdps)[:num_steps//2]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    X = current_poses[:, 0]
    Y = current_poses[:, 1]
    Z = current_poses[:, 2]
    U = achieved_delta_pose[:, 0]
    V = achieved_delta_pose[:, 1]
    W = achieved_delta_pose[:, 2]

    U2 = commanded_delta_pose[:, 0]
    V2 = commanded_delta_pose[:, 1]
    W2 = commanded_delta_pose[:, 2]
    '''

'''
#env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
env = RobotEnv('172.16.0.21', use_robot_cameras=True)
env.reset()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    #ax.scatter(X, Y, Z)
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0, 0.3])
    ax.set_zlim([0, 0.2])
    #plt.show()
    plt.savefig('images/plot2d.png')
    ax.quiver(X[1::10], Y[1::10], Z[1::10], U2[1::10], V2[1::10], W2[1::10], color='red')
    #plt.show()
    plt.savefig('images/plot2d_cdp.png')

    U3 = model_cdps[:, 0]
    V3 = model_cdps[:, 1]
    W3 = model_cdps[:, 2]
'''
