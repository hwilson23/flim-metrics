import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from tifffile import tifffile as tiff
import glob

print("running")
loc = ("G:\\")

filelist = []
photonlist = []
for files in glob.glob(loc +'*average_cvmap.tif'):
    filelist.append(files)
for files in glob.glob(loc + '*photons_cvmap.tif'):
    photonlist.append(files)
#print(filelist)
#print(photonlist)

for f in filelist:
    for p in photonlist:
        
        tmimage = np.array(tiff.imread(f"{loc}{f}"))

        tmimage = tmimage.astype(float)
        pimage = np.array(tiff.imread(f"{loc}{p}"))

        pimage = pimage.astype(float)


        fnum = np.divide(tmimage,pimage)
        tiff.imwrite(f'{loc}{f[:-4]}_fnum_correct.tif',fnum)
    
print("Done :)")