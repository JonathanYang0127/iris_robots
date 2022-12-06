import numpy as np
import pickle
import glob
import os
from iris_robots.transformations import add_angles, angle_diff

'''
TRAJECTORY_DIR = '/iris/u/jyang27/dev/iris_robot_learning/experiments/jonathan/purple_marker_grasp_franka'
glob_path = os.path.join(TRAJECTORY_DIR, '*.pkl')
files = glob.glob(glob_path)

trajectories = []
for f in files:
    with open(f, 'rb') as handle:
        trajectories.append(pickle.load(handle))
'''

DATA_PATH='/iris/u/jyang27/training_data/purple_marker_grasp_franka/combined_trajectories.npy'

with open(DATA_PATH, 'rb') as f:
    trajectories = np.load(DATA_PATH, allow_pickle=True)

print(len(trajectories))
def pose_diff(target, source):
    diff = np.zeros(len(target))
    diff[:3] = target[:3] - source[:3]
    diff[3:6] = angle_diff(target[3:6], source[3:6])
    diff[6] = target[6] - source[6]
    return diff

policy_actions = []
commanded_actions = []
real_actions = []
for t in trajectories:
    path = t['observations']
    for i in range(len(path) - 1):
        policy_actions.append(path[i][1])
        commanded_actions.append(path[i][2])
        real_actions.append(pose_diff(path[i+1][0]['desired_pose'], path[i][0]['desired_pose']))
        print(policy_actions[-1], real_actions[-1])

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
print("Mean squared error: {}".format(mean_squared_error(policy_actions, real_actions)))

