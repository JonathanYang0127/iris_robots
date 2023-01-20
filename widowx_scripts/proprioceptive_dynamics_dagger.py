import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from iris_robots.transformations import add_angles, angle_diff
import pickle

DATA_PATHS = [
        '/iris/u/jyang27/training_data/purple_marker_grasp_new/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/purple_marker_grasp_franka/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_blue_floral/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_mixed_floral/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_purple_marker_grasp_gray/combined_trajectories.npy',
        'proprioceptive_data_new/combined_trajectories.npy'
]
trajectories = []

for path in DATA_PATHS:
    trajectories.extend(np.load(path, allow_pickle=True))

actions = []
commanded_delta_pose = []        #next desired pose - current pose
achieved_delta_pose = []         #next achieved pose - current pose


def pose_diff(target, source):
    diff = np.zeros(len(target))
    diff[:3] = target[:3] - source[:3]
    diff[3:6] = angle_diff(target[3:6], source[3:6])
    diff[6] = target[6] - source[6]
    return diff


def limit_velocity(action):
    """Scales down the linear and angular magnitudes of the action"""
    action = np.array(action)
    lin_vel = action[:3]
    rot_vel = action[3:6]
    lin_vel_norm = np.linalg.norm(lin_vel)
    rot_vel_norm = np.linalg.norm(rot_vel)
    if lin_vel_norm > 1:
        lin_vel = lin_vel / lin_vel_norm
    if rot_vel_norm > 1:
        rot_vel = rot_vel / rot_vel_norm
    return np.concatenate((lin_vel, rot_vel))

 

for path in trajectories:
    for t in range(0, len(path['observations']) - 1):
        action = limit_velocity(path['actions'][t])
        current_pose = path['observations'][t]['current_pose']
        next_achieved_pose = path['observations'][t + 1]['current_pose']
        next_desired_pose = path['observations'][t + 1]['desired_pose']

        actions.append(action)
        adp = pose_diff(next_achieved_pose,  current_pose).tolist()[:-1]
        #adp = next_desired_pose[:-1]        

        for i in range(0, 2):
            index = 0 if t-i < 0 else t-i
            adp += path['observations'][index]['current_pose'].tolist()[:-1]
            adp += path['observations'][index]['desired_pose'].tolist()[:-1]
        #adp += path['observations'][t + 1]['current_pose'].tolist()
        
        #adp += (path['observations'][t]['current_pose']).tolist()
        #adp += (path['observations'][t - 1]['current_pose']).tolist() 
        achieved_delta_pose.append(adp)
        
        commanded_delta_pose.append(pose_diff(next_desired_pose, current_pose).tolist()[:-1])

commanded_delta_pose = np.array(commanded_delta_pose)
achieved_delta_pose = np.array(achieved_delta_pose)
actions = np.array(actions)

indices = np.arange(len(commanded_delta_pose)) 
indices = np.random.permutation(len(commanded_delta_pose))

num_train_indices = int(0.7 * len(commanded_delta_pose))
train_indices = indices[:num_train_indices]
test_indices = indices[num_train_indices:]

x_name = 'adp'
y_name = 'cdp'
x = achieved_delta_pose
y = commanded_delta_pose

print(x, y)
x_train = x[train_indices, :]
x_test = x[test_indices, :]
y_train = y[train_indices, :]
y_test = y[test_indices, :]

x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)


with open('action_normalization_mean.pkl', 'wb+') as f:
    pickle.dump((x_mean, x_std, y_mean, y_std), f)


normalize_name = 'unnormalized'
x_mean, x_std = np.zeros(x_train.shape[1]), np.ones(x_train.shape[1])
y_mean, y_std = np.zeros(y_train.shape[1]), np.ones(y_train.shape[1])


import pdb; pdb.set_trace()
x_train = (x_train - x_mean) / x_std 
x_test = (x_test - x_mean) / x_std
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: {}".format(r2_score(y_test, y_pred)))


with open('linear_cdp_model.pkl', 'wb+') as f: 
    pickle.dump(regr, f)

import torch
import torch.nn as nn
import torch.optim as optim
x_train_torch = torch.Tensor(x_train).cuda()
x_test_torch = torch.Tensor(x_test).cuda()
y_train_torch = torch.Tensor(y_train).cuda()
y_test_torch = torch.Tensor(y_test).cuda()
print(x_train.shape)


class InverseDynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(InverseDynamicsModel, self).__init__()
        modules = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        modules.extend([nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(self.dropout)])
        for i in range(3):
            modules.extend([nn.Linear(256, 256), nn.ReLU(), nn.Dropout(self.dropout)])
        modules.extend([nn.Linear(256 + input_dim, 256), nn.ReLU(), nn.Dropout(self.dropout)])
        for i in range(2):
            modules.extend([nn.Linear(256, 256), nn.ReLU(), nn.Dropout(self.dropout)])
        modules.extend([nn.Linear(256, output_dim), nn.ReLU()])
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for i in range(4 * 3):
            out = self.layers[i](out)
        out = torch.cat((out, x), dim=-1)
        for i in range(4 * 3, 8 * 3 - 1):
            out = self.layers[i](out)
        return out

model = nn.Sequential(
        nn.Linear(x_train.shape[1], 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 6)
        ).cuda()



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
            self.model_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/nonlinear_adp_cdp_model_'
            if normalize:
                self.model_path += 'normalized.pt'
            else:
                self.model_path += 'unnormalized.pt'
            with open(self.model_path, 'rb') as f:
                self.model = torch.load(f)

        self.normalization_path = '/iris/u/jyang27/dev/iris_robots/widowx_scripts/action_normalization_mean.pkl'
        with open(self.normalization_path, 'rb') as f:
            self.x_mean, self.x_std, self.y_mean, self.y_std = pickle.load(f)

        if not normalize:
            self.x_mean, self.x_std = np.zeros(self.x_mean.shape[0]), np.ones(self.x_std.shape[0])
            self.y_mean, self.y_std = np.zeros(self.y_mean.shape[0]), np.ones(self.y_std.shape[0])


    def postprocess_obs_action(self, obs, action):
        self._previous_obs = self._obs
        self._obs = obs
        adp = action.tolist()[:-1]
        adp += self._obs['current_pose'].tolist()[:-1]
        adp += self._obs['desired_pose'].tolist()[:-1]
        adp += self._previous_obs['current_pose'].tolist()[:-1]
        adp += self._previous_obs['desired_pose'].tolist()[:-1]
        adp = np.array(adp).reshape(1, -1)

        adp = (adp - self.x_mean) / self.x_std
        if self.model_type == 'linear':
            return self.model.predict(adp)[0]*self.y_std + self.y_mean
        elif self.model_type == 'nonlinear':
            adp = torch.Tensor(adp).cuda()
            return self.model(adp).detach().cpu().numpy()[0]*self.y_std + self.y_mean
                                                                                        

def collect_new_data():
    env = RobotEnv(robot_model='wx250s', control_hz=20, use_local_cameras=True, camera_types='cv2', blocking=False)
    X_data = []
    Y_data = []

    relabeller = DeltaPoseToCommand(obs, normalize=False, model_type='nonlinear')

    for i in range(20):
        index = np.random.randint(len(trajectories))
        obs = env.reset()
        for j in range(len(traj[index]['actions']) - 1):
            obs = env.get_observation()
            action = relabeller.postprocess_obs_action(obs, adp)
            action = np.concatenate((action, [0]))
            env.step_direct(action)
            new_obs = env.get_observation()
            X_data.append(pose_diff(new_obs['current_pose'], obs['current_pose']))
            Y_data.append(action)


#model = InverseDynamicsModel(x_train.shape[1], 7).cuda()
#import pdb; pdb.set_trace()
#model.load_state_dict(torch.load('new2/checkpoints_cdp_normalized_bigger/output_1100000.pt'))
import pdb; pdb.set_trace()
model = torch.load('nonlinear_adp_cdp_model_unnormalized.pt')
print(model(x_test_torch[0]).detach().cpu().numpy() * y_std + y_mean)
print(y_test_torch[0].detach().cpu().numpy() * y_std + y_mean)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)   
batch_size = 512

for i in range(100 * (int(1e4))):
    indices = torch.randint(x_train.shape[0], size=(batch_size,))
    optimizer.zero_grad()
    outputs = model(x_train_torch[indices])
    loss = criterion(outputs, y_train_torch[indices])
    loss.backward()
    optimizer.step()
    if (i % (int(1e4)) == 0):
        print("Train Loss: ", loss.item())
        outputs = model(x_test_torch)
        loss = criterion(outputs, y_test_torch)
        print("Validation Loss: ", loss.item())
        torch.save(model.state_dict(), 'new2/checkpoints_cdp_normalized_bigger/output_{}.pt'.format(i))
        torch.save(model, 'nonlinear_{}_{}_model_{}.pt'.format(x_name, y_name, normalize_name))
