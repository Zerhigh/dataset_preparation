from pathlib import Path
import pathlib
import pandas as pd
import geopandas as gpd
import shapely
import tqdm


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


# base = Path("U:/master/dl2/combined_download")
# base = Path("C:/Users/PC/Coding/cubexpress_austria/local_experiment/")
# base = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\combined_download')
base = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables')
file = base / 's2_ortho_download_data.csv'

all_data = []

for file in tqdm.tqdm(base.glob('*.csv')):
    name = file.stem
    data_ = pd.read_csv(file)
    cols = set(data_.columns)
    cols_keep = {'id', 'geometry', 'lat', 'lon'}
    cols = cols.difference(cols_keep)
    data_ = data_.drop(columns=list(cols))

    data_['geometry'] = data_.apply(to_point, axis=1)
    data = gpd.GeoDataFrame(data_, crs='EPSG:4326')
    data = data.to_crs(crs=31287)
    data['geometry'] = data.apply(to_square, axis=1)

    # data = data.to_crs(crs=4326)
    # data.to_file(str(base / f'{name}.shp'), driver='ESRI Shapefile')
    #data.to_file(str(base / f'wgs84_{name}.gpkg'), driver='GPKG')

    all_data.append(data)
    pass

gdf = pd.concat(all_data)
# gdf.to_file(str(base / f'wgs84_all_data.gpkg'), driver='GPKG')
gdf.to_file(str(base / f'MGI_all_data.shp'), driver='ESRI Shapefile')
pass
