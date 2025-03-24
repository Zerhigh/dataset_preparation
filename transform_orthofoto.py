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

ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full/')
sentinel2_base = Path('/data/USERS/shollend/sentinel2/full_austria/sr_inference/bilinear/')

ortho_trafo_target = ortho_base / 'target_transformed'
ortho_trafo_input = ortho_base / 'input_transformed'

ortho_trafo_target.mkdir(parents=True, exist_ok=True)
ortho_trafo_input.mkdir(parents=True, exist_ok=True)

download_table = pd.read_csv('/home/shollend/coding/download_stratified_ALL_S2_points_wdate_filter_combined.csv')
download_table_filtered = download_table[['id', 's2_download_id']]

austria = gpd.read_file('/data/USERS/shollend/oesterreich_border/oesterreich.shp')
austria32 = austria.to_crs('32632')
austria33 = austria.to_crs('32633')
geoms = {32: austria32.loc[[0], 'geometry'].values[0], 33: austria33.loc[[0], 'geometry'].values[0]}

statelog = pd.read_csv(ortho_base / 'statelog.csv')
statelog = statelog.merge(download_table_filtered, on='id', how='left')
statelog['in_austria'] = False


def transform_ortho(ortho_fp: str | Path,
                    ortho_op: str | Path,
                    s2_crs: str,
                    s2_transform: rio.transform.Affine,
                    s2_width: int,
                    s2_height: int) -> None:

    with rio.open(ortho_fp) as osrc:
        src_data = osrc.read()
        src_crs = osrc.crs

        dst_data = numpy.zeros(shape=(osrc.count, s2_width, s2_height), dtype='uint8')

        a, _ = reproject(
            source=src_data,
            destination=dst_data,
            src_transform=osrc.transform,
            src_crs=src_crs,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
            resampling=Resampling.nearest
        )

        profile = osrc.profile
        profile.update(transform=s2_transform, crs=s2_crs, width=s2_width, height=s2_height)

        with rio.open(ortho_op, 'w+', **profile) as dst:
            dst.write(dst_data)
    return


def _parallel(row: pd.Series) -> Tuple[int, bool]:
    s2_path = sentinel2_base / f'{row.s2_download_id}.tif'
    ortho_target = ortho_base / 'target' / f'target_{row.id}.tif'
    ortho_input = ortho_base / 'input' / f'input_{row.id}.tif'

    ex = [Path.exists(p) for p in (s2_path, ortho_target, ortho_input)]
    if False in ex:
        print(f'couldnt find file {s2_path, ortho_target, ortho_input}')
        return row.id, False

    # open sentinel image tile
    with rio.open(s2_path) as ssrc:
        dst_crs = ssrc.crs
        dst_transform = ssrc.transform
        dst_width, dst_height = ssrc.width, ssrc.height
        geom = box(*ssrc.bounds)

        border = geoms[dst_crs.data['zone']]

        if not geom.intersects(border):
            #print(f'tile {row.id} is out of bounds from austria')
            return row.id, False

    transform_ortho(ortho_fp=ortho_target, ortho_op=ortho_trafo_target / f'target_{row.id}.tif',
                    s2_crs=dst_crs,
                    s2_transform=dst_transform,
                    s2_width=dst_width,
                    s2_height=dst_height,
                    )

    transform_ortho(ortho_fp=ortho_input, ortho_op=ortho_trafo_input / f'input_{row.id}.tif',
                    s2_crs=dst_crs,
                    s2_transform=dst_transform,
                    s2_width=dst_width,
                    s2_height=dst_height,
                    )

    return row.id, True


def download_parallel():
    # Extract rows as dictionaries for easier parallel processing
    print('start')
    start = time.time()
    rows = [row for _, row in download_table.iterrows()]

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(_parallel, rows)

    for id, result in results:
        statelog.loc[statelog.id == id, 'in_austria'] = result
    statelog.to_csv(ortho_base / 'statelog_updated.csv')

    print(f'took: {round(time.time()-start, 2)}s')

    return

download_parallel()
