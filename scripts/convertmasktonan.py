from tifffile import tifffile as tiff
import numpy as np

img = np.array(tiff.imread("G:\\restitch_pdgfb_take3_ccv.tif"), dtype = float)
mask = np.array(tiff.imread("G:\\Nan_ovld_rmvd_restitch_pdgfb_take3.tif"))

mask[mask==float('NaN')] = 0
mask[mask>0] = 1

img_save = img*mask
img_save[img_save==0] = float('NaN')


tiff.imwrite("G:\\NaN_ovldrmd_ccv_pdgfb3_restitch_average.tif", img_save)
print("Done!!")