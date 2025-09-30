from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import jenkspy
import matplotlib.pyplot as plt

test_data = pd.read_csv(Path(r"C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\filtered\test.csv"))
cols_keep = ['id', 'assigned_class']
cols = set(test_data.columns).difference(cols_keep)
data = test_data.drop(columns=list(cols))

sr_data = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\tables\new_metrics')
all_metrics = pd.read_csv(sr_data / 'all_sr_model_results.csv')

tables = all_metrics.groupby('runname')

res_dict = {}

for i, table in tables:
    sr_name = i.split('_')[-1]
    print(sr_name)
    merged = pd.merge(data, table, left_on='id', right_on='image_id')

    vals = [x[1].drop(columns=['image_id', 'id', 'accuracy', 'runname']).mean().to_dict() for x in merged.groupby('assigned_class')]
    res = {}
    for val in vals:
        k = val.pop('assigned_class')
        res[k] = val
    res_dict[sr_name] = res

print(res_dict)
pass

# for table in sr_data.glob('full_*.csv'):
#     sr_name = table.name.split('_')[1].split('.')[0]
#     sr_data = pd.read_csv(table)
#     merged = pd.merge(data, sr_data, left_on='id', right_on='image_id')
#
#     # find all perfect scores
#     vals_no_mean = [x[1].drop(columns=['image_id', 'id', 'accuracy']) for x in merged.groupby('assigned_class')]
#     test = vals_no_mean[0]['iou'].to_list()
#     for x in test:
#         if x>0:
#             pass
#
#
#     vals = [x[1].drop(columns=['image_id', 'id', 'accuracy']).mean().to_dict() for x in merged.groupby('assigned_class')]
#     res = {}
#     for val in vals:
#         k = val.pop('assigned_class')
#         res[k] = val
#     res_dict[sr_name] = res
#
# print(res_dict)
# pass