# This takes a regions file as input and generates a 512x512 mask and saves it on at the given path

import json
from numpy import array, zeros
from scipy.misc import imsave
import sys

# verify the regions json path is given
if (len(sys.argv) < 3):
  print ('Usage: python generate_segments.py [PATH_TO_JSON] [OUTPUT_MASK_NAME]')
  exit(1)

# variable initialization
jsonPath = sys.argv[1]
print 'Loading regions from json at: ', jsonPath

# load the regions (training data only)
with open(jsonPath) as f:
    regions = json.load(f)

dims = (512, 512)

def tomask(coords):
    mask = zeros(dims)
    mask[zip(*coords)] = 1
    return mask

masks = array([tomask(s['coordinates']) for s in regions])
masks = masks.sum(axis=0)

imsave(sys.argv[2], masks)
