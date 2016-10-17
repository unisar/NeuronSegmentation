import json
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as mcolors
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

def outline(mask):
    horz = np.append(np.diff(mask,axis=1),np.zeros((mask.shape[0],1,mask.shape[2])),1)
    horz[horz!=0]==1
    vert = np.append(np.diff(mask,axis=2),np.zeros((mask.shape[0],mask.shape[1],1)),2)
    vert[vert!=0]==1
    r_horz = np.append(np.diff(mask[:,::-1,:],axis=1),np.zeros((mask.shape[0],1,mask.shape[2])),1)[:,::-1,:]
    r_horz[r_horz!=0]==1
    r_vert = np.append(np.diff(mask[:,:,::-1],axis=2),np.zeros((mask.shape[0],mask.shape[1],1)),2)[:,:,::-1]
    r_vert[r_vert!=0]==1
    comb = horz+vert+r_horz+r_vert
    comb[comb!=0]==1
    return comb

outlines = outline(masks)

#transparent colormap
colors = [(1,0,0,i) for i in np.linspace(0,1,3)]
cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

# show the outputs
fig = plt.figure(figsize=(30, 15))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

def update(i):
    ax1.clear()
    ax1.imshow(imgs[i], cmap='gray')
    ax1.set_title("frame %i" % i)
    ax2.clear()
    ax2.imshow(imgs[i], cmap='gray')
    ax2.imshow(outlines.sum(axis=0), cmap=cmap, vmin=0, vmax=1)
    ax2.set_title("frame %i" % i)
    
a = anim.FuncAnimation(fig, update, frames=len(imgs), repeat=False)
plt.show()