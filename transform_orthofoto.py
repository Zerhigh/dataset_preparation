import numpy
import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
from pathlib import Path

ortho = Path('C:/Users/PC/Coding/GeoQuery/demo/test_to_trafo/')
sentinel2 = Path('C:/Users/PC/Coding/s2inference/output/bilinear/')

for file in sentinel2.glob("*.tif"):
    ind = int(file.stem.split("_")[1])
    ortho_fn = f'input_{ind}.tif'

    with rio.open(file) as ssrc:
        dst_crs = ssrc.crs  # Get CRS of raster 2
        dst_transform = ssrc.transform  # Get transform (extent & resolution)
        dst_width, dst_height = ssrc.width, ssrc.height  # Get dimensions

    with rio.open(ortho / ortho_fn) as osrc:
        src_data = osrc.read()  # Read first band
        src_crs = osrc.crs

        # Create a new raster with same shape as raster 2
        dst_data = numpy.zeros(shape=(4, ssrc.shape[1], ssrc.shape[1]), dtype='uint8')

        a, _ = reproject(
            source=src_data,
            destination=dst_data,
            src_transform=osrc.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest  # You can change to nearest, cubic, etc.
        )

        print(np.unique(dst_data))

        profile = osrc.profile
        profile.update(transform=dst_transform, crs=dst_crs, width=dst_width, height=dst_height)

        with rio.open(f'output_{ind}.tif', 'w+', **profile) as dst:
            dst.write(dst_data)

    pass
    break