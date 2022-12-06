import numpy as np
import torch
import pickle
import argparse

from iris_robots.transformations import add_angles, angle_diff
import rlkit.torch.pytorch_util as ptu

ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/purple_marker_grasp_new/combined_trajectories.npy'
ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/purple_marker_grasp_franka/combined_trajectories.npy'
#MODEL_PATH = '/iris/u/jyang27/logs/22-11-28-BC-wx250/22-11-28-BC-wx250_2022_11_28_21_57_58_id000--s5/itr_1000.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/22-11-28-BC-wx250/22-11-28-BC-wx250_2022_11_28_21_58_14_id000--s6/itr_1000.pt'
MODEL_PATH = '/iris/u/jyang27/logs/22-11-29-BC-wx250/22-11-29-BC-wx250_2022_11_29_13_49_40_id000--s0/itr_1000.pt'

with open(ROBOT_PATH_ONE, 'rb') as f:
    traj1 = np.load(f, allow_pickle=True)

with open(ROBOT_PATH_TWO, 'rb') as f:
    traj2 = np.load(f, allow_pickle=True)


img1 = traj1[11]['observations'][10]['images'][0]['array'].astype(np.float32)/ 255.0
img2 = traj2[11]['observations'][10]['images'][0]['array'].astype(np.float32) / 255.0
img3 = 0.5 * img1 + 0.5 * img2

import matplotlib.pyplot as plt
plt.axis('off')
plt.imshow(img1)
plt.show()

