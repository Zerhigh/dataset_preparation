import time
from typing import Tuple

import numpy
import os
import numpy as np
import pandas as pd
import shutil
from multiprocessing import Pool
from tqdm import tqdm
import geopandas as gpd
import rasterio as rio
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
from pathlib import Path

# files = Path('/data/USERS/shollend/taco_example/hr_mask/')
# files = Path('/data/USERS/shollend/taco/hr_mask/')
#files = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses/target/')
files = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses/target_transformed/')

land_use_inverted = {
    1: 41,
    2: 83,
    3: 59,
    4: 60,
    5: 61,
    6: 64,
    7: 40,
    8: 48,
    9: 57,
    10: 55,
    11: 56,
    12: 58,
    13: 42,
    14: 62,
    15: 63,
    16: 65,
    17: 72,
    18: 84,
    19: 87,
    20: 88,
    21: 92,
    22: 95,
    23: 96,
    24: 52,
    25: 54
}


def _parallel(raster_path):
    # Open the raster file
    with rio.open(raster_path, 'r+') as src:
        raster_data = src.read(1)  # Read first band

        # Apply value mapping
        for old_value, new_value in land_use_inverted.items():
            raster_data[raster_data == old_value] = new_value

        # Write back to the same file
        src.write(raster_data, 1)


def process_parallel():
    # Extract rows as dictionaries for easier parallel processing
    print('start')
    start = time.time()
    rows =  [rp for rp in files.glob('*.tif')]

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(_parallel, rows)


process_parallel()