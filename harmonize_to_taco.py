import time
from traceback import print_tb
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

ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses/')
sentinel2_base = Path('/data/USERS/shollend/sentinel2/full_austria/output_s2/')
ortho_target_base = ortho_base / 'target_transformed'
ortho_input_base = ortho_base / 'input_transformed'

taco = Path('/data/USERS/shollend/taco_example2/')
ts2 = taco / 'lr_s2'
tmask = taco / 'hr_mask'
tortho = taco / 'hr_orthofoto'

ts2.mkdir(parents=True, exist_ok=True)
tmask.mkdir(parents=True, exist_ok=True)
tortho.mkdir(parents=True, exist_ok=True)

statelog = pd.read_csv(ortho_base / 'statelog_updated.csv')


def harmoinze(df):
    # drop tiles outside of austria
    df = df[df.in_austria == True].copy()
    df['image_id'] = ''

    new_data = {}

    for i, (id, row) in tqdm(enumerate(df.iterrows())):
        s2 = sentinel2_base / f'{row.s2_download_id}.tif'
        ortho_target = ortho_target_base / f'target_{row.id}.tif'
        ortho_input = ortho_input_base / f'input_{row.id}.tif'

        new_id = f"{i:05d}"
        row.image_id = f"{new_id}.tif"
        shutil.copy(s2, ts2 / f'S2_{new_id}.tif')
        shutil.copy(ortho_target, tmask / f'HR_mask_{new_id}.tif')
        shutil.copy(ortho_input, tortho / f'HR_ortho_{new_id}.tif')

        new_data[i] = row

    new_df = pd.DataFrame.from_dict(new_data, orient="index")

    try:
        new_df.rename(columns={"id": "orthofoto_id"}, inplace=True)
        new_df.drop(columns=["Unnamed: 0", "contains_nodata", "aerial", "cadaster"], inplace=True, errors="ignore")
    except:
        print('error renaming')

    new_df.to_csv(taco / 'metadata.csv', index=False)


    return


harmoinze(df=statelog)
