import numpy as np
import matplotlib.pyplot as plt
PATH = '/iris/u/jyang27/training_data/wx250_shelf_close_camera2/wx250_shelf_close_grasp_camera2/combined_trajectories.npy'
PATH = '/iris/u/jyang27/dev/iris_robots/iris_robots/training_data/wx250_pickplace_final/wx250_pickplace_black_marker_tan/Sat_Sep__2_16:20:12_2023.npy'
PATH = '/iris/u/jyang27/training_data/wx250_shelf_close_camera2/wx250_shelf_close_grasp_new_compartment_reverse_camera2/Sat_Sep__2_15:28:29_2023.npy'

with open(PATH, 'rb') as f:
    data = np.load(f, allow_pickle=True)

if 'combined' in PATH:
    data = data[0]
else:
    data = data.item()

for image in data['observations'][0]['images']:
    plt.figure()
    plt.imshow(image['array'])
    plt.show()
