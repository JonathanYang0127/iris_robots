import numpy as np
import argparse
import pickle
import os
from pathlib import Path

def combine_trajectories(args):
    files = []
    for d in args.directories:
        files.extend(Path(d).rglob('*.npy'))
        files.extend(Path(d).rglob('*.pkl'))
    print(files)
    trajectories = []
    for f in files:
        file = str(f)
        if 'combined' not in file:
            with open(file, 'rb') as f:
                if 'npy' in file:
                    trajectory = np.load(f, allow_pickle=True)
                    trajectory = trajectory.item()
                else:
                    trajectory = pickle.load(f)

            if not args.ignore_checks: 
                closed_pose = False
                for j in range(len(trajectory['actions'])):
                    if trajectory['actions'][j][6] > 0.5:
                        closed_pose = True
                        break
            
            if args.ignore_checks:
                trajectories.append(trajectory)
            else: 
                if trajectory['observations'][0]['images'] == []:
                    print("No image in {}".format(file))
                elif not closed_pose:
                    print("No grasp in {}".format(file))
                else:
                    trajectories.append(trajectory)

    save_path = os.path.join(args.save_dir, 'combined_trajectories.npy')
    print("Saving {} combined trajectories to {}".format(len(trajectories), save_path))
    np.save(os.path.abspath(save_path), trajectories)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directories", type=str, nargs='+', required=True)
    parser.add_argument("-s", "--save_dir", type=str, default=None)
    parser.add_argument("-i", "--ignore-checks", action="store_true", default=False)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = args.directories[0]

    combine_trajectories(args)
