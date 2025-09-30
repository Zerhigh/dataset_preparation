# this script takes a csv and a number of images (already INFERRED) as input, matching the ids in the images,
# it will add the geotransforms and crs as columns to the csv

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import jenkspy
import matplotlib.pyplot as plt

# tu = False
# if tu:
#     s2data = Path('/data/USERS/shollend/sentinel2/sr_inference/bilinear/')
# else:
#     s2data = Path('./data/')
#


def assign_class(fg_ratio):
    if fg_ratio == 0:
        return 0
    elif fg_ratio < 0.01:
        return 1
    elif fg_ratio < 0.05:
        return 2
    elif fg_ratio < 0.2:
        return 3
    else:
        return 4


def get_col_statistics(df: gpd.GeoDataFrame, col: str) -> None:
    print(f'statistics {col}:')
    print(f'    mean: {round(df[col].mean(), 3)}')
    print(f'    median: {round(df[col].median(), 3)}')
    return


def get_class_statistics(df: gpd.GeoDataFrame, col: str) -> None:
    print(f'class statistics {col}:')
    class_counts = df[col].value_counts().sort_index()
    for i, val in class_counts.items():
        print(f'    {i}: {val}, {round(val/len(df) * 100, 2)}%')
    return


def assign_jenks_class(ratio, jenks):
    for i in range(1, len(jenks)):
        if jenks[i - 1] <= ratio <= jenks[i]:
            return i - 1
    return -1


def filter_df(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # filter 1: NoData Hard boolean filter
    df_rej = df[df['contains_nodata'] == True]
    df_rej.to_file(f'{output_dir}/nodata_tiles.gpkg', driver='GPKG')
    df = df[df['contains_nodata'] == False]

    # filter 1: NoData from 1% onwards
    # df_rej = df[df['nodata_total'] >= 0.01]
    # df_rej.to_file('stratification_tests/nodata_tiles.gpkg', driver='GPKG')
    # df = df[df['nodata_total'] < 0.01]

    print(f'Number of filtered tiles (NoData): {len(df)}')

    # filter 2: Reduce number of NoBuilding tiles
    meta_buildings = df[df['dist_41'] > 0]
    meta_no_buildings = df[df['dist_41'] == 0]

    print(f'Number of tiles with buildings (before readding): {len(meta_buildings)}')

    # num of samples to have 5% of total sample number non-buildings
    percent = 0.05
    num_rows = int((percent / (1 - percent)) * len(meta_buildings))

    # sample these no-building tiles
    sampled_no_buildings = meta_no_buildings.sample(n=num_rows, random_state=32)
    building_focused_df = pd.concat([meta_buildings, sampled_no_buildings], axis=0)
    print(f'Number of filtered tiles (NoBuildings): {len(building_focused_df)}')

    return building_focused_df


def train_split(df: gpd.GeoDataFrame, col: str, split: List[float] = [0.7, 0.15, 0.15]) -> List[gpd.GeoDataFrame]:
    train_ratio = split[0]
    val_ratio = split[1]
    test_ratio = split[2]

    train, temp = train_test_split(
        df, test_size=(1 - train_ratio), stratify=df[col], random_state=39
    )

    # Then, split temp into validation and test sets
    val, test = train_test_split(
        temp, test_size=(test_ratio / (test_ratio + val_ratio)),
        stratify=temp[col], random_state=39
    )

    # train.to_csv('train_tables/train_ortho.csv', index=False)
    # test.to_csv('train_tables/test_ortho.csv', index=False)
    # val.to_csv('train_tables/val_ortho.csv', index=False)

    return [train, test, val]


def stratify_jenks(df: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, list]:
    # force a 0 class into the breaks
    breaks = [0]
    for v in jenkspy.jenks_breaks(df["dist_41"].to_list(), n_classes=5):
        breaks.append(v)

    df['assigned_class'] = df["dist_41"].apply(assign_jenks_class, args=(breaks,))

    return df, breaks


def plot_histogram(df, col='assigned_class'):
    n_classes = df[col].nunique()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(df['dist_41'], bins=n_classes, alpha=0.7, edgecolor='black')

    plt.title('Histogram of dist_41 Stratified by Jenks Classes')
    plt.xlabel('dist_41 (% building pixels)')
    plt.ylabel('Frequency')
    plt.legend(title='Jenks Class')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


BASE = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\combined_download')
output_dir = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\combined_download_correlated_testing')
tiles = gpd.read_file(BASE / 's2_ortho_download_data.gpkg')

Path('stratification_tests').mkdir(parents=True, exist_ok=True)

print('------------------------------')
print(f'Number of tiles: {len(tiles)}')

filtered_tiles = filter_df(df=tiles)
stratified_tiles, breaks = stratify_jenks(filtered_tiles)

#plot_histogram(stratified_tiles)
print('------------------------------')
print('Class Breaks:')
for i in range(1, len(breaks)):
    print(f'    {breaks[i - 1]} - {breaks[i]}')

print('------------------------------')
print('Statistics: Full Dataset')
get_col_statistics(df=stratified_tiles, col='dist_41')
get_col_statistics(df=stratified_tiles, col='nodata_total')
get_class_statistics(df=stratified_tiles, col='assigned_class')

print('------------------------------')
train, test, val = train_split(df=stratified_tiles, col='assigned_class')
print('Datasplits:')
print(f'    train: {len(train)}')
print(f'    test: {len(test)}')
print(f'    val: {len(val)}')

print('------------------------------')
print('Statistics: Train')
get_col_statistics(df=train, col='dist_41')
get_col_statistics(df=train, col='nodata_total')
get_class_statistics(df=train, col='assigned_class')

print('------------------------------')
print('Statistics: Test')
get_col_statistics(df=test, col='dist_41')
get_col_statistics(df=test, col='nodata_total')
get_class_statistics(df=test, col='assigned_class')

print('------------------------------')
print('Statistics: Validate')
get_col_statistics(df=val, col='dist_41')
get_col_statistics(df=val, col='nodata_total')
get_class_statistics(df=val, col='assigned_class')

print('------------------------------')

filtered_tiles.to_file(f'{output_dir}/filtered_no_buildings.gpkg', driver='GPKG')
stratified_tiles.to_file(f'{output_dir}/filtered_no_buildings_stratified.gpkg', driver='GPKG')

train.to_file(f'{output_dir}/train.gpkg', driver='GPKG')
test.to_file(f'{output_dir}/test.gpkg', driver='GPKG')
val.to_file(f'{output_dir}/val.gpkg', driver='GPKG')

train.to_csv(f'{output_dir}/train.csv')
test.to_csv(f'{output_dir}/test.csv')
val.to_csv(f'{output_dir}/val.csv')




