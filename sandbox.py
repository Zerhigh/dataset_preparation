import shutil
from pathlib import Path
import pathlib
import pandas as pd
import geopandas as gpd
import shapely
import tqdm
import shapely

base = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data')
file = base / 'stratification_tables' / 'wgs84_all_data_subset_graz.gpkg'
all_data = base / 'combined_download' / 's2_ortho_download_data.gpkg'


def prepare_cropped_area():
    gdf = gpd.read_file(str(file))
    vals = gdf.geometry.total_bounds
    bbox2 = shapely.box(*vals)
    # bbox_gdf = gpd.GeoDataFrame(index=)
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[bbox2])
    bbox_gdf_ = bbox_gdf.to_crs(crs=31287)

    all_data_file = gpd.read_file(all_data)
    intersection = all_data_file[all_data_file.intersects(bbox_gdf_.geometry.iloc[0])]
    # intersection = all_data_file.sjoin(bbox_gdf_, how='right')

    # bbox_gdf.to_file(str(base / f'bbox_subset_graz.shp'), driver='ESRI Shapefile')
    intersection.to_file(str(base / f'intersection.gpkg'), driver='GPKG')



subset_graz = r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\graz_subset\austrian_shps_subsets\bbox_subset_graz_tiles.shp"
download_info = Path(r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\graz_subset\intersection_s2_downlaod_info.gpkg")
s2_tiles = Path(r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\combined_download\lr_s2")
to = Path(r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\graz_subset\files")

gdf = gpd.read_file(subset_graz)
gdf['id'] = gdf['id'].astype('int')
metadata = gpd.read_file(download_info)
cols = set(metadata.columns)
cols_keep = {'s2_id', 's2_full_id', 'id'}
cols = cols.difference(cols_keep)
data_ = metadata.drop(columns=list(cols))

gdf_wmeta = gdf.merge(data_, on='id')
gdf_wmeta = gdf_wmeta.drop(columns=['fid'])
gdf_wmeta.to_file(r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\graz_subset\austrian_shps_subsets\selection_wmetadata.shp", driver='ESRI Shapefile')

s2_ids = set(gdf_wmeta['s2_id'])


for iid in gdf['id']:
    id = f'S2_{int(iid):5d}.tif'
    s2_file = s2_tiles / id
    if s2_file.exists():
        shutil.copy(s2_file, to / id)
    else:
        raise FileNotFoundError(s2_file)

    pass

pass