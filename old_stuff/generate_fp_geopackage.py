import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import pathlib
from typing import Dict
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import pyproj

from shapely.geometry import Point
from shapely.ops import transform


if __name__ == "__main__":
    # define paths for image and upsampled sentinel2 images
    sentinel2_base = Path('/data/USERS/shollend/sentinel2/full_austria/sr_inference/bilinear/')
    ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses/')

    ROOT_DIR = Path("/data/USERS/shollend/taco")
    df = pd.read_csv(ROOT_DIR / "metadata_updated.csv")

    res = []

    to_epsg = "EPSG:31287"
    rows = [row for _, row in df.iterrows()]

    def _parallel(row):
        with rio.open(row["hr_othofoto_path"], 'r') as src:
            geom = box(*src.bounds)

            project = pyproj.Transformer.from_crs(src.crs, to_epsg, always_xy=True).transform
            trafo_bounds = transform(project, geom)

            res = {'geometry': trafo_bounds,
                        "image_id": row["image_id"],
                        "hr_mask_path": row["hr_mask_path"],
                        "hr_compressed_mask_path": row["hr_compressed_mask_path"],
                        "hr_othofoto_path": row["hr_othofoto_path"],
                        "lr_s2_path": row["lr_s2_path"],
                        "lr_harm_path": row["lr_harm_path"],
                        "hr_harm_path": row["hr_harm_path"],
                        "tortilla_path": row["tortilla_path"],
                        "low_corr": row["low_corr"],
                        "cs_cdf": row["cs_cdf"]
                        }

        return res

    with Pool(processes=os.cpu_count()) as pool:
        #results = pool.map(_parallel, rows)
        results = list(tqdm(pool.imap_unordered(_parallel, rows), total=len(rows), desc="Processing"))

    gdf = gpd.GeoDataFrame(results, crs=to_epsg)
    gdf.to_file("image_footprints.gpkg", driver='GPKG')

    pass



    # merge
    # statelog = statelog.merge(download_table_filtered, on='id', how='left')
    # statelog = statelog.merge(ortho, on='ARCHIVNR', how='left')