


# Brief chronology
- uw-loci/ MultivariateAnalysis_FLIM
  - 2025 Wilson et al. SPIE Proceedings Paper   
- uw-loci/ flim-metrics  (FLIM_variability_simulations)
- uw-loci/ flim-metrics-viewer (hwilson23/ python_threshold_overlays)
- uw-loc/ flim-metrics-multiplexing (hwilson23/ multiplexing_VAE)


# Workflow description 
for FLIM quality metric visualization as a means to validate consistency across large-area non-homogeneous FLIM datasets, Wilson et al. 

Imaging
With tissue on scope, generate xy grid in MicroManager with create grid MDA. Manually replace some xy points with added z-focused positions (to create xyz coordinates). Export position list .pos
Use interpolate-z-coords.py to to create xyz coordinate file (.csv)
Run main MM_bigfovacqusition tiled-acquisition with .csv file that has interpolated z-plane positions
This produces a tile_config.txt that has position list and image coordinates 
Script will save .sdt, .spc, .json files 

Analysis
FLIM analysis was completed with SPCImage software (version 8.9, Becker & Hickl, Berlin, Germany). For tissues, we used biexponential fitting with maximum likelihood estimation and a spatial bin value of three (corresponding to 7 x 7 pixel binning). The shift value was set to 0 for all images after the respective IRF was imported for each tissue.
This was completed with batch fitting in SPCImage
Creates .img, chi.asc, color_Image.tif, colorcodedvalue.asc (lifetime image), photons.asc
Git bash used to remove space in “color coded value” files
asc_to_tiff.ijm used to create tif files for chi, colorcodedvalue, photons
To prepare for stitching:
horizontal_filp_tiff.ijm (bc stage coordinate sign doesn’t match)
scale tile config to proper units scale_tile_config.py
Since .tif not saved, used spc files with same overload criteria to determine if file overloaded spc_conversion.py (recorded the overload positive files in a text file)
any overloaded files were removed from tileconfig list remove_ovld_file.py
Use grid stitcher in Fiji to stitch files (chi, colorcodedvalue, photons) by just changing file extension in tileconfig
Linear blending used for photon images (and chi), average used for lifetime
Image processing <large-stitched-image> <computational-image>:
triangle threshold run to remove background - background and overloaded pixels set to zero or nan to be excluded from post processing
convertmasktonan.py     used to mask out both 0 and NaN
Create CV map for both lifetime and photon images using create_cvmap_convolution.py (version with generic filter), kernel size set to 49x49
Produces binned image (so you can see kernel average) and cvmap
F value images created with create_fnum_image.py or also just division in Fiji

User Interface <image>:
Load desired image for masking (e.g. lifetime or F value image), set masking limits and export masks
These masks can be used to create data graphs (ex. create_boxplot_foruimasking.ipynb)

ROI subset selection
To identify which images were selected in tissue ROI, used an image number stitched image with same pixel size as regular so ROI could be transferred 
Number images created with makenumberimages.m 
Stitched again with grid stitcher and tile config to make sure images in right place
Used Fiji to copy ROI and get list of unique image pos numbers featured in ROI, text matching used to create respective tileconfig file for ROI only then grid stitcher used to create image
Images then used for analysis, ex. create_boxplot_tumorarearandom.ipynb

Wasserstein Data  
Phasor: matched ROI images put into separate folder for GS lab processing, phasor (G,S) data exported, wasserstein info calculated with tissueROi_wassersteindist.ipynb
Regular flim graph used same code (with for loops for stdev) but edited for phasor

Other figure parts
Histograms for corresponding images created with histograms_images.ipynb
Box plots like create_boxplot_direct_alltissue.ipynb
Strip plots create_fnumstripplot.ipynb


# FLIM_varability_simulation
Biexponential Phasor Simulations
Tiff files with desired settings (lifetimes, fractional components, photons) created with biexpestimate.ipynb (or could use the shortened version with just functions) (flim-metrics)
Tiff images loaded to GSLab and phasor info exported
Analysis with computingoverlaphull.ipynb

- biexp_simulations.py is python version of jupyter notebook
- biexpestimate_shortened_py.py is the function only for use with the pypi project
- working version biexpestimate.ipynb

Varability code demonstrated on Fig2 of 
FLIM Quality Metric Visualization as a means to validate consistency across large-area non-homogeneous FLIM datasets
Wilson et al.
