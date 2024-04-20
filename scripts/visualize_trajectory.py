import numpy as np
import pickle
import argparse
from PIL import Image


PATH = ''

with open(PATH, 'rb') as f:
    data = np.load(PATH, allow_pickle=True)
    data = data.item()

images = []
for obs in data['observations']:
    image = obs['images'][0]['array']
    images.append(Image.fromarray(image))

images[0].save('trajectory.gif', save_all=True,optimize=False, append_images=images[1:], loop=0)
