# example python script for loading neurofinder data
#
# for more info see:
#
# - http://neurofinder.codeneuro.org
# - https://github.com/codeneuro/neurofinder
#
# requires three python packages
#
# - numpy
# - scipy
# - matplotlib
#

import json
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from numpy import array, zeros
from scipy.misc import imread
from glob import glob
import numpy as np

# load the images
files = sorted(glob('images/*.tiff'))
imgs = array([imread(f) for f in files])
dims = imgs.shape[1:]

# load the regions (training data only)
with open('regions/regions.json') as f:
    regions = json.load(f)

def tomask(coords):
    mask = zeros(dims)
    mask[zip(*coords)] = 1
    return mask

masks = array([tomask(s['coordinates']) for s in regions])

# show the outputs
fig = plt.figure(figsize=(30, 15))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(masks.sum(axis=0), cmap='gray')

def update(i):
    ax1.clear()
    ax1.imshow(imgs[i], cmap='gray')
    ax1.set_title("frame %i" % i)
    
a = anim.FuncAnimation(fig, update, frames=len(imgs), repeat=False)
plt.show()