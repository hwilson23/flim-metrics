import json
import numpy as np
import tifffile


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def export_decay_data_as_3d_tiff(
    times,
    counts,
    filename="decay_data_3d.tiff",
    normalize=False,
    time_dimension_first=True,
    box_size=(40, 40),
):
    times = np.asarray(times)
    counts = np.asarray(counts)
    width, height = box_size
    if time_dimension_first:
        single_slice = np.ones((width, height))
        data_3d = np.zeros((len(counts), width, height), dtype=float)
        for t in range(len(counts)):
            data_3d[t, :, :] = single_slice * counts[t]
    else:
        data_3d = np.zeros((width, height, len(counts)), dtype=float)
        for x in range(width):
            for y in range(height):
                data_3d[x, y, :] = counts
    if normalize:
        min_val = np.min(data_3d)
        max_val = np.max(data_3d)
        if max_val > min_val:  # Prevent division by zero
            data_3d = ((data_3d - min_val) / (max_val - min_val) * 65535).astype(
                np.uint16
            )
        else:
            data_3d = np.zeros_like(data_3d, dtype=np.uint16)
    blankarr = np.zeros((256, 256, 256))
    print(data_3d.shape)
    blankarr[
        :,
        127 - box_size[0] // 2 : 127 + box_size[0] // 2,
        127 - box_size[1] // 2 : 127 + box_size[1] // 2,
    ] = data_3d
    data_3d = blankarr
    metadata = {
        "time_points": times,
        "time_unit": "nanoseconds",
        "min_time": times[0],
        "max_time": times[-1],
        "time_step": times[1] - times[0] if len(times) > 1 else 0,
        "box_size": box_size,
        "description": "Decay curve repeated across a 2D image",
    }
    metadata_str = json.dumps(metadata, cls=NumpyEncoder)
    tifffile.imwrite(
        filename,
        data_3d,
        metadata={"time_info": metadata_str},
        imagej=False,  # Make it compatible with ImageJ/Fiji
        resolution=(1.0, 1.0),  # Placeholder resolution
        photometric="minisblack",
        compression="lzw",  # Lossless compression
    )
    print(f"Decay data exported as 3D TIFF with dimensions {data_3d.shape}: {filename}")

    # Also save a simple CSV with the raw decay curve data
    csv_filename = filename.replace(".tiff", ".csv").replace(".tif", ".csv")
    np.savetxt(
        csv_filename,
        np.column_stack((times, counts)),
        delimiter=",",
        header="Time(ns),Counts",
        comments="",
    )

    print(f"Raw decay data saved as CSV: {csv_filename}")

    return filename


def export_decay_3d_withnoise(
    times,
    decay3dblock,
    filename="decay_data_3d.tiff",
    normalize=False,
    time_dimension_first=True,
    box_size=(40, 40),
):
    data_3d = decay3dblock
    if normalize:
        min_val = np.min(data_3d)
        max_val = np.max(data_3d)
        if max_val > min_val:  # Prevent division by zero
            data_3d = ((data_3d - min_val) / (max_val - min_val) * 65535).astype(
                np.uint16
            )
        else:
            data_3d = np.zeros_like(data_3d, dtype=np.uint16)
    blankarr = np.zeros((256, 256, 256))
    print(data_3d.shape)
    blankarr[
        :,
        127 - box_size[0] // 2 : 127 + box_size[0] // 2,
        127 - box_size[1] // 2 : 127 + box_size[1] // 2,
    ] = data_3d
    data_3d = blankarr
    metadata = {
        "time_points": times,
        "time_unit": "nanoseconds",
        "min_time": times[0],
        "max_time": times[-1],
        "time_step": times[1] - times[0] if len(times) > 1 else 0,
        "box_size": box_size,
        "description": "Decay curve repeated across a 2D image",
    }
    metadata_str = json.dumps(metadata, cls=NumpyEncoder)
    tifffile.imwrite(
        filename,
        data_3d,
        metadata={"time_info": metadata_str},
        imagej=False,  # Make it compatible with ImageJ/Fiji
        resolution=(1.0, 1.0),  # Placeholder resolution
        photometric="minisblack",
        compression="lzw",  # Lossless compression
    )
    print(f"Decay data exported as 3D TIFF with dimensions {data_3d.shape}: {filename}")
    return filename
