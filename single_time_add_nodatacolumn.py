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
from datetime import datetime

ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses/')
statelog = pd.read_csv(ortho_base / 'statelog_updated_statistics.csv')
print(statelog.columns)
ftaco = Path('/data/USERS/shollend/taco/metadata_updated.csv')

taco = pd.read_csv(ftaco)

statelog.drop(columns=['id', 'aerial', 'cadaster', 'dist_0', 'dist_40',
       'dist_41', 'dist_42', 'dist_48', 'dist_52', 'dist_54', 'dist_55',
       'dist_56', 'dist_57', 'dist_58', 'dist_59', 'dist_60', 'dist_61',
       'dist_62', 'dist_63', 'dist_64', 'dist_65', 'dist_72', 'dist_83',
       'dist_84', 'dist_87', 'dist_88', 'dist_92', 'dist_95', 's2_full_id',
       'lon', 'lat', 'cs_cdf', 'corine', 'abs_days_diff', 'time', 'ARCHIVNR', 'in_austria', 'ortho_begin_date', 'ortho_end_date',
       'dist_96'], inplace=True)

statelog.rename(columns={"contains_nodata": "orthofoto_contains_nodata"}, inplace=True)

test = taco.merge(statelog, on='s2_download_id', how='left')
test.to_csv('/data/USERS/shollend/taco/metadata_updated.csv')

