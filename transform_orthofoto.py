import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict
from rasterio.warp import reproject, Resampling
from shapely.geometry import box


def convert_date(date_str: str) -> datetime:
    """
    Convert a German date string to a datetime object.

    :param date_str: The input date string in 'dd-Mon-yy' format with German month names.
    :return: A datetime object.
    :raises ValueError: If the date string is incorrectly formatted.
    """
    try:
        day, month, year = date_str.split("-")
        german_to_english_months = {
            "Jan": "Jan", "Feb": "Feb", "Mär": "Mar", "Apr": "Apr", "Mai": "May",
            "Jun": "Jun", "Jul": "Jul", "Aug": "Aug", "Sep": "Sep", "Okt": "Oct",
            "Nov": "Nov", "Dez": "Dec"
        }

        month = german_to_english_months.get(month, month)
        return datetime.strptime(f"{day}-{month}-{year}", "%d-%b-%y")
    except Exception as e:
        raise ValueError(f"Error parsing date {date_str}: {e}")


def transform_ortho(ortho_fp: str | Path,
                    ortho_op: str | Path,
                    s2_crs: str,
                    s2_transform: rio.transform.Affine,
                    s2_width: int,
                    s2_height: int) -> None | Dict:

    # open orthofoto image tile
    with rio.open(ortho_fp) as osrc:
        src_data = osrc.read()
        src_crs = osrc.crs

        # define empt arrar and reproject data into it
        dst_data = np.zeros(shape=(osrc.count, s2_width, s2_height), dtype='uint8')

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

        # write data to file
        with rio.open(ortho_op, 'w+', **profile) as dst:
            dst.write(dst_data)

        # if its the mask, recalculate the statitics
        if osrc.count == 1:
            sats = {}
            # update metadata count etc
            num_px = s2_width * s2_height
            # counts cannot be set as its rasterized!
            instance_counts = None #gdf['label'].value_counts().to_dict()

            for label in labels_and_nodata:
                count = np.count_nonzero(dst_data == label)

                sats[f'dist_{label}'] = round(count / num_px, 3)
                sats[f'count_{label}'] = instance_counts

            return sats
        else:
            return None


def _parallel(row: pd.Series) -> pd.Series:
    # define individual image paths
    s2_path = sentinel2_base / f'{row.s2_download_id}.tif'
    ortho_target = ortho_base / 'target' / f'target_{row.id}.tif'
    ortho_input = ortho_base / 'input' / f'input_{row.id}.tif'

    # check if everything exists
    ex = [Path.exists(p) for p in (s2_path, ortho_target, ortho_input)]
    if False in ex:
        print(f'couldnt find file {s2_path, ortho_target, ortho_input}')
        return row

    # open sentinel image tile
    with rio.open(s2_path) as ssrc:
        dst_crs = ssrc.crs
        dst_transform = ssrc.transform
        dst_width, dst_height = ssrc.width, ssrc.height
        geom = box(*ssrc.bounds)

        # get border transformed to corresponding image crs
        border = geoms[dst_crs.data['zone']]

        if not geom.intersects(border):
            print(f'tile {row.id} is out of bounds from austria')
            return row
        else:
            # transform and resample the mask
            stats = transform_ortho(ortho_fp=ortho_target, ortho_op=ortho_trafo_target / f'target_{row.id}.tif',
                            s2_crs=dst_crs,
                            s2_transform=dst_transform,
                            s2_width=dst_width,
                            s2_height=dst_height,
                            )

            # transform and resample the true image
            _ = transform_ortho(ortho_fp=ortho_input, ortho_op=ortho_trafo_input / f'input_{row.id}.tif',
                            s2_crs=dst_crs,
                            s2_transform=dst_transform,
                            s2_width=dst_width,
                            s2_height=dst_height,
                            )

            # reapply the newly calculated statistics
            for index, value in stats.items():
                row.loc[index] = value

            # set attribute value to hint tile is in asutria
            row['in_austria'] = True
            return row


def download_parallel() -> pd.DataFrame:
    # Extract rows as dictionaries for easier parallel processing
    print('start')
    start = time.time()
    rows = [row for _, row in statelog.iterrows()]

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(_parallel, rows)

    new_df = pd.DataFrame(results)

    print(f'took: {round(time.time()-start, 2)}s')

    return new_df


def download_sequential() -> pd.DataFrame:
    # Extract rows as dictionaries for easier parallel processing
    print('start')
    start = time.time()

    results = []
    for _, row in statelog.iterrows():
        results.append(_parallel(row))

    new_df = pd.DataFrame(results)

    print(f'took: {round(time.time()-start, 2)}s')

    return new_df


if __name__ == "__main__":
    # define paths for image and upsampled sentinel2 images
    ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full_allclasses/')
    sentinel2_base = Path('/data/USERS/shollend/sentinel2/full_austria/sr_inference/bilinear/')

    # define outpaths and create directories
    ortho_trafo_target = ortho_base / 'target_transformed_statistics'
    ortho_trafo_input = ortho_base / 'input_transformed_statistics'
    ortho_trafo_target.mkdir(parents=True, exist_ok=True)
    ortho_trafo_input.mkdir(parents=True, exist_ok=True)

    # get statelog from downloading and merge with download additioanl metadata
    statelog = pd.read_csv(ortho_base / 'statelog.csv')
    statelog['in_austria'] = False

    # load table used for downloading data to get metadata for taco
    download_table = pd.read_csv('/home/shollend/coding/download_stratified_ALL_S2_points_wdate_filter_combined.csv')
    download_table_filtered = download_table[
        ['id', 's2_full_id', 'lon', 'lat', 'cs_cdf', 'corine', 'abs_days_diff', 'time', 'ARCHIVNR', 's2_download_id']]

    # load austrian borders to better test if points are inside austria and ahve accoriding values
    austria = gpd.read_file('/data/USERS/shollend/oesterreich_border/oesterreich.shp')
    austria32 = austria.to_crs('32632')
    austria33 = austria.to_crs('32633')
    geoms = {32: austria32.loc[[0], 'geometry'].values[0], 33: austria33.loc[[0], 'geometry'].values[0]}

    # load metadata file to access begin and end dates of caputirng of orthofotos for taco
    fortho = Path('/data/USERS/shollend/metadata/matched_metadata.gpkg')
    ortho = gpd.read_file(fortho)
    ortho.drop(columns=['Operat', 'Jahr', 'Date', 'prevTime', 'vector_url', 'RGB_raster', 'NIR_raster', 'geometry'],
               inplace=True)
    ortho.rename(columns={"beginLifeS": "ortho_begin_date", 'endLifeSpa': 'ortho_end_date'}, inplace=True)
    ortho['ortho_begin_date'] = ortho['ortho_begin_date'].apply(convert_date)
    ortho['ortho_end_date'] = ortho['ortho_end_date'].apply(convert_date)

    # merge
    statelog = statelog.merge(download_table_filtered, on='id', how='left')
    statelog = statelog.merge(ortho, on='ARCHIVNR', how='left')

    # define available labels for filtering after resampling
    labels_and_nodata = (
    0, 40, 41, 42, 48, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72, 83, 84, 87, 88, 92, 95, 96)

    # execute resampling and new metadata creation
    #df = download_sequential()
    df = download_parallel()

    df.drop(columns=[f'count_{i}' for i in labels_and_nodata], inplace=True)
    df.to_csv(ortho_base / 'statelog_updated_statistics.csv', index=False)
