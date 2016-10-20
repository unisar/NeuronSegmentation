'''
Saves numpy array of max pixel values across all images, padded by 16 on each side
Also saves the mask that indicates which pixels belong to ROI, padded by 16 on each side
'''

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy import array, zeros
from scipy.misc import imread
from glob import glob
import numpy as np
import os

# load the images
files = sorted(glob('images/*.tiff'))
imgs = array([imread(f) for f in files])
dims = imgs.shape[1:]

#generate mean intensity pixels over time
sum = np.sum(imgs,axis=0)
mean = np.mean(sum)
std = np.std(sum)
print "min:",np.amin(sum)
print "max:",np.amax(sum)
print "mean:",mean
print "std:",std
sum[sum > (mean + 3*std)] = mean + 3*std   #clip pixels above 99 percentile intensity
sum = sum/mean

#save numpy array
path = os.getcwd().split("\\")[-1]
np.save("X_"+path,sum)

# show the outputs
plt.figure()
plt.imshow(sum, cmap='gray')
plt.show()