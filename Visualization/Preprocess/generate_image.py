from glob import glob
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy import array, zeros
from scipy.misc import imread
import numpy as np
from scipy.misc import imsave

# verify the regions json path is given
if (len(sys.argv) < 3):
  print ('Usage: python generate_image.py [PATH_TO_IMAGES_FOLDER] [OUTPUT_IMAGE_NAME]')
  exit(1)

# load the images
files = sorted(glob(sys.argv[1] + '*.tiff'))
imgs = array([imread(f) for f in files])

sum = np.sum(imgs,axis=0)
mean = np.mean(sum)
std = np.std(sum)
sum[sum > (mean + 3*std)] = mean + 3*std

imsave(sys.argv[2], sum)
