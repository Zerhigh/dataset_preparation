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

ftaco = Path('/data/USERS/shollend/metadata/metadata_taco.csv')
fortho = Path('/data/USERS/shollend/metadata/matched_metadata.gpkg')

taco = pd.read_csv(ftaco)
ortho = gpd.read_file(fortho)
ortho.drop(columns=['Operat', 'Jahr', 'Date', 'prevTime', 'vector_url', 'RGB_raster', 'NIR_raster', 'geometry'], inplace=True)
ortho.rename(columns={"beginLifeS": "ortho_begin_date", 'endLifeSpa': 'ortho_end_date'}, inplace=True)

ortho['ortho_begin_date'] = ortho['ortho_begin_date'].apply(convert_date)
ortho['ortho_end_date'] = ortho['ortho_end_date'].apply(convert_date)
print(ortho)
test = taco.merge(ortho, on='ARCHIVNR', how='left')
test.to_csv('/data/USERS/shollend/metadata/metadata_taco_wdate.csv')

