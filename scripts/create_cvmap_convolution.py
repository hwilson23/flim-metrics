import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from tifffile import tifffile as tiff
import glob

print("running")
kernel_size = [7,7] 
loc = ("G:\\MNtissueproject\\A_MN_figures\\processed_images\\")

filelist = []
for files in glob.glob(loc +'*refit_average_correct.tif'):
    filelist.append(files)
print(filelist)

for f in filelist:
    print(f)
    image = np.array(tiff.imread(f"{loc}{f}"))

    image = image.astype(float)
        
    # Get image dimensions
    height, width = image.shape

        
    meanfilter = generic_filter(image, np.nanmean, kernel_size)#ignores nan values
    stdfilter = generic_filter(image, np.nanstd, kernel_size)#ignores nan values

    cv_map = np.divide(stdfilter,meanfilter)
   


    #tiff.imwrite(f'{loc}{f[:-5]}_bin{kernel_size[1]}_cvmap.tif',cv_map)
    #tiff.imwrite(f'{loc}{f[:-5]}_bin{kernel_size[1]}_binnedmeanmap.tif',meanfilter)
    tiff.imwrite(f'G:\\MNtissueproject_CLEANED20250716\\figures\\testsmallCVkernal\\{f[:-5]}_bin{kernel_size[1]}_cvmap.tif',cv_map)
    tiff.imwrite(f'G:\\MNtissueproject_CLEANED20250716\\figures\\testsmallCVkernal\\{f[:-5]}_bin{kernel_size[1]}_binnedmeanmap.tif',meanfilter)

    #plt.imshow(result)
    plt.imshow(cv_map)
    plt.clim(0,np.percentile(cv_map, 10))
    plt.colorbar()
    #plt.show()
    plt.title(f'CV Map {f}')

print("Done :)")