import time
from typing import Tuple

import numpy
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import geopandas as gpd
import rasterio as rio
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
from pathlib import Path

ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses/')
sentinel2_base = Path('/data/USERS/shollend/sentinel2/full_austria/sr_inference/bilinear/')

ortho_target = ortho_base / 'target_transformed'
ortho_input = ortho_base / 'input_transformed'

statelog = pd.read_csv(ortho_base / 'statelog_updated.csv')


def harmoinze():

    return