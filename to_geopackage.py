from pathlib import Path
import pathlib
import pandas as pd
import geopandas as gpd
import shapely

def to_point(row):
    return shapely.Point(row['lon'], row['lat'])


def to_square(row, w=640):
    lons = [row['geometry'].x - 640,
            row['geometry'].x + 640,
            row['geometry'].x + 640,
            row['geometry'].x - 640,
            row['geometry'].x - 640, ]
    lats = [row['geometry'].y + 640,
            row['geometry'].y + 640,
            row['geometry'].y - 640,
            row['geometry'].y - 640,
            row['geometry'].y + 640, ]
    return shapely.Polygon(list(zip(lons, lats)))

base = Path("U:/master/dl2/combined_download")
base = Path("C:/Users/PC/Coding/cubexpress_austria/local_experiment/")
base = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\combined_download')
file = base / 's2_ortho_download_data.csv'


data_ = pd.read_csv(file)
data_['geometry'] = data_.apply(to_point, axis=1)
data = gpd.GeoDataFrame(data_, crs='EPSG:4326')
data = data.to_crs(crs=31287)
data['geometry'] = data.apply(to_square, axis=1)
data.to_file(base / 's2_ortho_download_data.gpkg', driver='GPKG')
pass
