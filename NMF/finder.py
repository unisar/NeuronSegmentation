import json
from numpy import zeros, random, asarray, where, ones
from scipy.ndimage.morphology import binary_closing, binary_dilation
import thunder as td
from regional import many
import matplotlib.pyplot as plt
import numpy as np
from extraction import NMF
import neurofinder


'''
This is based upon the Local NMF example
from:  https://github.com/thunder-project/thunder-extraction/blob/master/extraction/algorithms/nmf.py

Instead of using all the raw images as it is, I have preprocessed the images
using median_filters and changed the block size for images when applying NMF
'''
class finder(object):

    def __init__(self,file_names,no_of_files):
        self.file_names =  file_names
        self.no_of_files = no_of_files


    def read_images(self,file_path):
        print ("Reading images from: %s" % file_path)
        images = td.images.fromtif(file_path + '/images', ext='tiff', stop=self.no_of_files)
        imgs = images.median_filter(size=3)
        filtered_imgs = imgs.toarray()
        m2 = np.mean(filtered_imgs,axis=0)
        return imgs, filtered_imgs, m2

    def read_regions(self,file_path):
        print ("Reading regions of: %s" % file_path)
        with open(file_path + '/regions/regions.json', 'r') as f:
            regions = json.load(f)
        
        #list of lists of coordinates
        regions_list = many([region['coordinates'] for region in regions])
        return regions_list
   
    def plot_original_predicted(self,fig_1, fig_2):
        fig = plt.figure(figsize=(30, 15))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.imshow(fig_1)
        ax2.imshow(fig_2)
        plt.show()

    def mask_image(self,regions,imgs,color):
        masks = regions.mask(stroke=color,fill=None,base=imgs.clip(0,2000)/2000)
        return masks

    def applyNMF(self,isPlot, isTrain):
        results = []
        for f in self.file_names:    
            imgs,filtered_imgs, sumImgs  = self.read_images(f)

            print ("Applying NMF to: %s" % f)
            algorithm = NMF(k=5, percentile=99, max_iter=100)
            model = algorithm.fit(filtered_imgs,chunk_size=(30,30),padding=(25,25))
            merged = model.merge(0.1)
            nmask = self.mask_image(merged.regions,sumImgs,'blue')

            if isTrain:
                regions = self.read_regions(f)
                omask =  self.mask_image(regions,sumImgs,'red')
                print neurofinder.centers(regions, merged.regions)
                print neurofinder.shapes(regions, merged.regions)
                if isPlot:
                    self.plot_original_predicted(omask,nmask)

            print('found %g regions' % merged.regions.count) 
            coords = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
            result = {'dataset': f, 'regions': coords}
            results.append(result)
            print('')

        return results


def main():

    test_files = [
                    './neurofinder.00.00.test','./neurofinder.00.01.test','./neurofinder.01.00.test',
                    './neurofinder.01.01.test','./neurofinder.02.00.test','./neurofinder.02.01.test',
                    './neurofinder.03.00.test','./neurofinder.04.00.test','./neurofinder.04.01.test'
                ]
    n = finder(test_files,None)
    results = n.applyNMF(False, False)
    with open('output.json', 'w') as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    main()      








    


