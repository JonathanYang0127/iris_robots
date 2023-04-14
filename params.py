import numpy as np

#Robot-Specific Parameters
ROBOT_PARAMS = dict(
    {'franka': dict({
        'num_joints': 7,
        'reset_joints': np.array([0., -0.24, 0, - 9 / 10 * np.pi, 0,  5 / 6 * np.pi, np.pi / 4])
    }),
    'wx200': dict({
        'num_joints': 5,
        'reset_joints': np.array([0, -0.5, 0.5, np.pi / 2, 0.])
    }),
    'wx250s': dict({
        'num_joints': 6,
        'reset_joints': np.array([0, -0.3, 0.3, 0, np.pi / 2, 0])
    }),
    'fetch': dict({
        'num_joints': 7,
        'reset_joints': np.array([0, -1.0, 0, 1.5, 0,  1.2, 0])
    )}
})
