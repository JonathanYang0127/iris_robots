import numpy as np
import torch
import pickle
import argparse

import rlkit.torch.pytorch_util as ptu
from rlkit.misc.wx250_utils import (add_multitask_data_to_singletask_buffer_real_robot,
    add_multitask_data_to_multitask_buffer_real_robot, DummyEnv,
    configure_dataloader_params)
from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer


#ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_blue_nodesired_control3/combined_trajectories.npy'
#ROBOT_PATH_ONE = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_tan_nodesired_control3/combined_trajectories.npy'
#ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_gray_red_cup_nodesired_control3/combined_trajectories.npy'
#ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/franka_nodesired/franka_black_marker_grasp_gray_nodesired/combined_trajectories.npy'
#ROBOT_PATH_TWO = ROBOT_PATH_ONE
#ROBOT_PATH_TWO = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_blue_nodesired_control3/combined_trajectories.npy'
#MODEL_PATH = '/iris/u/jyang27/logs/23-02-22-BC-wx250/23-02-22-BC-wx250_2023_02_22_15_58_09_id000--s2/itr_1400.pt'
MODEL_PATH= '/iris/u/jyang27/logs/23-02-24-BC-wx250/23-02-24-BC-wx250_2023_02_24_14_11_37_id000--s2/itr_340.pt'
#MODEL_PATH = '/iris/u/jyang27/logs/23-02-24-BC-wx250/23-02-24-BC-wx250_2023_02_24_14_11_37_id000--s2/itr_1400.pt'
MODEL_PATH = '/iris/u/jyang27/logs/23-02-24-BC-wx250/23-02-24-BC-wx250_2023_02_24_14_11_37_id000--s2/itr_1000.pt'
#TEST_ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_floral_mixed_cloth_nodesired_control3/combined_trajectories.npy'
#TEST_ROBOT_PATH = '/iris/u/jyang27/training_data/franka_nodesired/franka_black_marker_grasp_pink_polka_plate_nodesired/combined_trajectories.npy'
#TEST_ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_gray_nodesired_control3_ee2/combined_trajectories.npy'
TEST_ROBOT_PATH = '/iris/u/jyang27/training_data/wx250_nodesired_control3_ee2/wx250_measuring_cup_grasp_nodesired_control3_ee2/combined_trajectories.npy'

variant = dict(
    image_size=64,
    use_robot_state=True,
    dataloader_params=dict(
        stack_frames=False,
        action_relabelling=None,
        downsample_image=True,
        align_actions=True,
        mixup=True,
        continuous_to_blocking=False,
    ),
    buffers = ['/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_tan_nodesired_control3/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_gray_red_cup_nodesired_control3/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_nodesired_control3/wx250_black_marker_grasp_blue_nodesired_control3/combined_trajectories.npy',]
)
configure_dataloader_params(variant)

env = DummyEnv(image_size=variant['image_size'], task_embedding_dim=0)
if variant['use_robot_state']:
    observation_keys = ['image', 'desired_pose', 'current_pose', 'task_embedding']
    state_observation_dim = sum([env.observation_space.spaces[k].low.size for k in observation_keys if
        k != 'image' and k != 'task_embedding'])
else:
    observation_keys = ['image', 'task_embedding']
    state_observation_dim = 0
buffer_kwargs = dict(
    use_next_obs_in_context=False,
    sparse_rewards=False,
    observation_keys=observation_keys[:-1],
    internal_keys=['mixup_distance']
)
replay_buffer = ObsDictMultiTaskReplayBuffer(
    int(1E6),
    env,
    np.arange(1),
    **buffer_kwargs,
)

for b in variant['buffers']:
    buffer_params = {0: b}
    add_multitask_data_to_multitask_buffer_real_robot(buffer_params, replay_buffer,
        task_encoder=None, embedding_mode='None',
        encoder_type='image')


replay_buffer.task_buffers[0].precompute_mixup_sample_indices(threshold=0.15)

import pdb; pdb.set_trace()
with open(MODEL_PATH, 'rb') as f:
    params = torch.load(f)
    policy = params['evaluation/policy']

#policy.output_conv_channels = False
policy.color_jitter = False
policy.feature_norm = False

anchor_batch, positive_batch, negative_batch = \
    replay_buffer.random_anchor_positive_and_negative_batches(0, 2000)

ao, po, no = ptu.from_numpy(anchor_batch["observations"]).cuda(), \
    ptu.from_numpy(positive_batch["observations"]).cuda(), ptu.from_numpy(negative_batch["observations"]).cuda()

task_buffer = replay_buffer.task_buffers[0]
ai, pi, ni = task_buffer.get_mixup_distance(anchor_batch["indices"]), \
    task_buffer.get_mixup_distance(positive_batch["indices"]), \
    task_buffer.get_mixup_distance(negative_batch["indices"])

output1 = policy.forward(ao, intermediate_output_layer=2)[1]
output2 = policy.forward(po, intermediate_output_layer=2)[1]
output3 = policy.forward(no, intermediate_output_layer=2)[1]

loss_func = torch.nn.TripletMarginLoss(margin=1.0, p=2)
loss = loss_func(output1, output2, output3)
print(loss)

import matplotlib.pyplot as mpl
ao_test = ao.cpu().numpy()
po_test = po.cpu().numpy()
no_test = no.cpu().numpy()
import matplotlib.pyplot as plt
for i in range(20):
    index = np.random.randint(256)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(ao_test[index][:3*64*64].reshape(3, 64, 64).transpose(1, 2, 0))
    ax[1].imshow(po_test[index][:3*64*64].reshape(3, 64, 64).transpose(1, 2, 0))
    ax[2].imshow(no_test[index][:3*64*64].reshape(3, 64, 64).transpose(1, 2, 0))
    plt.savefig('images/out{}.png'.format(i))
    plt.close(fig)
    print(ai[index])
    print(pi[index])
    print(ni[index])
    print(torch.linalg.norm(output1[index] - output2[index]), torch.linalg.norm(output1[index] - output3[index]))
    print(" ")

test_buffer = ObsDictMultiTaskReplayBuffer(
    int(1E6),
    env,
    np.arange(1),
    **buffer_kwargs,
)

buffer_params = {0: TEST_ROBOT_PATH}
add_multitask_data_to_multitask_buffer_real_robot(buffer_params, test_buffer,
    task_encoder=None, embedding_mode='None',
    encoder_type='image')


batch = test_buffer.random_batch(0, 2000)
closest_batch = ptu.from_numpy(batch['observations']).cuda()
output4 = policy.forward(closest_batch, intermediate_output_layer=2)[1]
closest_test = closest_batch.cpu().numpy()


for i in range(20):
    nearest = None
    nearest_dist = 1e9
    for index in range(2000):
        if torch.linalg.norm(output1[i] - output4[index]) < nearest_dist:
            nearest_dist = torch.linalg.norm(output1[i] - output4[index])
            nearest = closest_test[index][:3*64*64].reshape(3, 64, 64).transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(ao_test[i][:3*64*64].reshape(3, 64, 64).transpose(1, 2, 0))
    ax[1].imshow(nearest)
    plt.savefig('images/out_nearest{}.png'.format(i))
    plt.close(fig)
