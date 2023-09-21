import numpy as np
import torch
import pickle
import argparse

import rlkit.torch.pytorch_util as ptu
from rlkit.misc.wx250_utils import (add_multitask_data_to_singletask_buffer_real_robot,
    add_multitask_data_to_multitask_buffer_real_robot, DummyEnv,
    configure_dataloader_params)
from rlkit.data_management.multitask_replay_buffer import ObsDictMultiTaskReplayBuffer


variant = dict(
    image_size=64,
    use_robot_state=True,
    dataloader_params=dict(
        stack_frames=False,
        action_relabelling=None,
        downsample_image=True,
        align_actions=True,
        mixup=True,
        continuous_to_blocking=False
    ),
)

buffers1 = [
        #'/iris/u/jyang27/training_data/franka_shelf_close/franka_shelf_close_camera3/combined_trajectories.npy',
        '/iris/u/jyang27/training_data/wx250_shelf_close_camera2/wx250_shelf_close_grasp_camera2/combined_trajectories.npy',
        #'/iris/u/jyang27/training_data/wx250_shelf_close_camera2/wx250_shelf_close_grasp_reverse_camera2/combined_trajectories.npy'
]
buffers2 = [
    '/iris/u/jyang27/training_data/franka_shelf_close/franka_shelf_close_camera3/combined_trajectories.npy'
]


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
    internal_keys=['mixup_distance', 'robot_id']
)
replay_buffer1 = ObsDictMultiTaskReplayBuffer(
    int(1E5),
    env,
    np.arange(1),
    **buffer_kwargs,
)

replay_buffer2 = ObsDictMultiTaskReplayBuffer(
    int(1E5),
    env,
    np.arange(1),
    **buffer_kwargs,
)

for b in buffers1:
    buffer_params = {0: b}
    add_multitask_data_to_multitask_buffer_real_robot(buffer_params, replay_buffer1,
        task_encoder=None, embedding_mode='None',
        encoder_type='image')

for b in buffers2:
    buffer_params = {0: b}
    add_multitask_data_to_multitask_buffer_real_robot(buffer_params, replay_buffer2,
        task_encoder=None, embedding_mode='None',
        encoder_type='image')


index = 0
start = replay_buffer1.task_buffers[0]._path_starts[index]
end = replay_buffer1.task_buffers[0]._path_ends[index]
md1 = replay_buffer1.task_buffers[0]._obs['mixup_distance'][start:end]
pose1 = replay_buffer1.task_buffers[0]._obs['current_pose'][start:end]


index = 0
start = replay_buffer2.task_buffers[0]._path_starts[index]
end = replay_buffer2.task_buffers[0]._path_ends[index]
md2 = replay_buffer2.task_buffers[0]._obs['mixup_distance'][start:end]
pose2 = replay_buffer2.task_buffers[0]._obs['current_pose'][start:end]

print(pose1[0:5])
print(pose2[0:5])
import matplotlib.pyplot as plt

distance = md1[30]
distances1 = np.linalg.norm((md2 - distance)[:,0:3], axis=-1)
distances = np.linalg.norm((md2 - distance)[:,3:6], axis=-1)
for i in range(len(md2)):
    print(distances1[i], distances[i])

