import numpy
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import rasterio as rio
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
from pathlib import Path

ortho_base = Path('/data/USERS/shollend/orthophoto/austria_full/')
sentinel2_base = Path('/data/USERS/shollend/sentinel2/full_austria/sr_inference/bilinear/')

ortho_trafo_target = ortho_base / 'target_transformed2'
ortho_trafo_input = ortho_base / 'input_transformed2'

ortho_trafo_target.mkdir(parents=True, exist_ok=True)
ortho_trafo_input.mkdir(parents=True, exist_ok=True)

download_table = pd.read_csv('/home/shollend/coding/download_stratified_ALL_S2_points_wdate_filter_combined.csv')
austria = gpd.read_file('/data/USERS/shollend/oesterreich_border/oesterreich.shp')
austria32 = austria.to_crs('32632')
austria33 = austria.to_crs('32633')
border32 = austria32.loc[[0], 'geometry'].values[0]
border33 = austria33.loc[[0], 'geometry'].values[0]

geoms = {32: border32, 33: border33}

def transform_ortho(ortho_fp: str | Path,
                    ortho_op: str | Path,
                    s2_crs: str,
                    s2_transform: rio.transform.Affine,
                    s2_width: int,
                    s2_height: int):

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

excluded_ortho = []
excluded_s2 = []

for i, row in tqdm(download_table.iterrows()):
    s2_path = sentinel2_base / f'{row.s2_download_id}.tif'
    ortho_target = ortho_base / 'target' / f'target_{row.id}.tif'
    ortho_input = ortho_base / 'input' / f'input_{row.id}.tif'

    ex = [Path.exists(p) for p in (s2_path, ortho_target, ortho_input)]
    if False in ex:
        print(f'couldnt find file {s2_path, ortho_target, ortho_input}')
        excluded_ortho.append(f'target_{row.id}.tif')
        excluded_s2.append(f'{row.s2_download_id}.tif')
        continue

    # open sentinel image tile
    with rio.open(s2_path) as ssrc:
        dst_crs = ssrc.crs
        dst_transform = ssrc.transform
        dst_width, dst_height = ssrc.width, ssrc.height
        geom = box(*ssrc.bounds)

        border = geoms[dst_crs.data['zone']]

        if not geom.intersects(border):
            print(f'tile {row.id} is out of bounds from austria')
            continue

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

print(excluded_s2)
print(excluded_ortho)

#
# for file in sentinel2_base.glob("*.tif"):
#     ind = int(file.stem.split("_")[1])
#     ortho_fn = f'input_{ind}.tif'
#
#     with rio.open(file) as ssrc:
#         dst_crs = ssrc.crs  # Get CRS of raster 2
#         dst_transform = ssrc.transform  # Get transform (extent & resolution)
#         dst_width, dst_height = ssrc.width, ssrc.height  # Get dimensions
#
#     with rio.open(ortho_base / ortho_fn) as osrc:
#         src_data = osrc.read()  # Read first band
#         src_crs = osrc.crs
#
#         # Create a new raster with same shape as raster 2
#         dst_data = numpy.zeros(shape=(4, ssrc.shape[1], ssrc.shape[1]), dtype='uint8')
#
#         a, _ = reproject(
#             source=src_data,
#             destination=dst_data,
#             src_transform=osrc.transform,
#             src_crs=src_crs,
#             dst_transform=dst_transform,
#             dst_crs=dst_crs,
#             resampling=Resampling.nearest  # You can change to nearest, cubic, etc.
#         )
#
#         print(np.unique(dst_data))
#
#         profile = osrc.profile
#         profile.update(transform=dst_transform, crs=dst_crs, width=dst_width, height=dst_height)
#
#         with rio.open(f'output_{ind}.tif', 'w+', **profile) as dst:
#             dst.write(dst_data)
#
#     pass
#     break