import numpy as np
import json
import tifffile

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def simulate_multi_exponential_decay(num_photons, lifetimes, fractions, time_range, time_bins=256, background=0):
    """
    Simulate a multi-exponential fluorescence decay curve.
    
    Parameters:
    -----------
    num_photons : int
        Total number of photons to simulate
    lifetimes : list of float
        Lifetimes in nanoseconds for each component
    fractions : list of float
        Relative fractions for each component (will be normalized)
    time_range : tuple
        (min_time, max_time) in nanoseconds
    time_bins : int
        Number of time bins to use
    background : float
        Expected background counts per bin
        
    Returns:
    --------
    times : numpy array
        Time points of the simulation
    counts : numpy array
        Photon counts at each time point
    """

    min_time, max_time = time_range
    
    # Normalize fractions to sum to 1
    norm_fractions = np.array(fractions) / np.sum(fractions)

    # Determine number of photons for each component
    component_photons = np.random.multinomial(num_photons, norm_fractions)
    #print(f'component_photons: {component_photons}')
    
    # Initialize empty array for all photon arrival times
    all_photon_times = np.array([])
    
    # Generate photons for each lifetime component
    for i, (lifetime, n_photons) in enumerate(zip(lifetimes, component_photons)):
        if n_photons > 0:
            # Generate arrival times for this component
            photon_times = np.random.exponential(scale=lifetime, size=n_photons)
            all_photon_times = np.append(all_photon_times, photon_times)
            #print(f"len all photons:{len(all_photon_times)}")
    # Filter out photons outside the time range
    #print(f'num photons out of time range: {sum(all_photon_times > max_time)}')
    all_photon_times = all_photon_times[all_photon_times <= max_time]

    
    # Create the binned histogram
    bin_edges = np.linspace(min_time, max_time, time_bins + 1)
    counts, _ = np.histogram(all_photon_times, bins=bin_edges)
    
    # Calculate bin centers for time axis
    times = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Add random background counts if specified
    if background > 0:
        bg_counts = np.random.poisson(background, size=time_bins)
        counts = counts + bg_counts
        #print(f'total counts without background: {sum(counts-bg_counts)}')
    
    return times, counts

def export_decay_3d_withnoise(times,decay3dblock, filename="decay_data_3d.tiff",
                               normalize=False, time_dimension_first=True, box_size=(40, 40)):
    data_3d = decay3dblock
   
    # Normalize to 16-bit range if requested
    if normalize:
        # Scale to 0-65535 (full 16-bit range)
        min_val = np.min(data_3d)
        max_val = np.max(data_3d)
        if max_val > min_val:  # Prevent division by zero
            data_3d = ((data_3d - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        else:
            data_3d = np.zeros_like(data_3d, dtype=np.uint16)
   
    #put decay into empty 256x256
    #blankarr = np.zeros((256,256,256))
    #print(data_3d.shape)
    #blankarr[:,127-box_size[0]//2:127+box_size[0]//2,127-box_size[1]//2:127+box_size[1]//2] =data_3d
    #data_3d = blankarr

    # Save metadata about the time axis
    metadata = {
        'time_points': times,
        'time_unit': 'nanoseconds',
        'min_time': times[0],
        'max_time': times[-1],
        'time_step': times[1] - times[0] if len(times) > 1 else 0,
        'box_size': box_size,
        'description': 'Decay curve repeated across a 2D image'
    }
   
    # Convert metadata to string for embedding in TIFF tags
    metadata_str = json.dumps(metadata, cls=NumpyEncoder)
   
    # Save as TIFF with metadata
    tifffile.imwrite(
        filename,
        data_3d,
        metadata={'time_info': metadata_str},
        imagej=False,  # Make it compatible with ImageJ/Fiji
        resolution=(1.0, 1.0),  # Placeholder resolution
        photometric='minisblack',
        compression='lzw'  # Lossless compression
    )
   
    print(f"Decay data exported as 3D TIFF with dimensions {data_3d.shape}: {filename}") 
    
    return filename

if __name__ == "__main__":
    ###Input Settings
    # Create a new biexponential decay example with known parameters
    num_photons = 10000
    num_photons = [500,2000,10000,10000,2000,500]
    true_lifetimes = [1.1,0]  # Two lifetime components (in ns)
    true_fractions = [1,0]  # Relative fractions (70% and 30%)
    time_range = (0, 12.5)  # nanoseconds
    time_bins = 256
    #background_level = (num_photons*0.2)/256 #background counts per bin
    background_level = 0
    width = 8
    height = 8

    min_time, max_time = time_range
    bin_edges = np.linspace(min_time, max_time, time_bins + 1)
    times = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    data_block = np.zeros((len(times), width, height), dtype=float)
    boxsize = width*height
    #true_lifetimes_1 = np.logspace(np.log10(1),np.log10(4),4,endpoint=True)
    true_lifetimes_1 = np.logspace(np.log10(1),np.log10(4),3,endpoint=True)
    
    true_lifetimes_1 = np.append(true_lifetimes_1, (true_lifetimes_1))
    
    #true_lifetime_2 = np.logspace(np.log10(0.5),np.log10(2),2,endpoint=True)
    true_lifetime_2 = 0.5
    #true_fractions_1 = np.linspace(0,0.7,3,endpoint=True)
    true_fractions_1 = 1

    ##run
    x = 0
    y = 0
    
    completeimage = np.zeros((len(times), width*len(true_lifetimes_1),height*len(num_photons)))
   
    for n in num_photons:
    
        for l1 in true_lifetimes_1:
            #for l2 in true_lifetime_2:
            l2 = true_lifetime_2
                #for a1 in true_fractions_1:
            a1 = true_fractions_1
            for i in range(width):
                for j in range(height):
                    multi_times, data_block[:,i,j] = simulate_multi_exponential_decay(
                        num_photons=n,
                        lifetimes=(l1,l2),
                        fractions=(a1,1-a1),
                        time_range=time_range,
                        time_bins=time_bins,
                        background=background_level
                        )
                    
            print(f'file: t1_{l1}_t2_{l2}_a1_{a1}_a2_{1-a1}')
            print("x,y:", x, y, "data_block:", data_block.shape)
            completeimage[:,x:(x+8),y:(y+8)] = data_block
            x = x+8
        x = 0
        y = y+8
print(np.shape(completeimage))         
export_decay_3d_withnoise(multi_times,completeimage,f"G:\\MNtissueproject_CLEANED20250716\\biexp_test\\MULTI_withnoise_np_500_t1_{np.round(l1,2)}_t2_{np.round(l2,2)}_a1_{np.round(a1,2)}_a2_{np.round((1-a1),2)}_data.tif", box_size=(width*len(true_lifetimes_1),height*len(num_photons)))
'''
    for l1 in true_lifetimes_1:
        for l2 in true_lifetime_2:
            for a1 in true_fractions_1:
                for i in range(width):
                    for j in range(height):
                        multi_times, data_block[:,i,j] = simulate_multi_exponential_decay(
                            num_photons=num_photons,
                            lifetimes=(l1,l2),
                            fractions=(a1,1-a1),
                            time_range=time_range,
                            time_bins=time_bins,
                            background=background_level
                            )
'''

    