from pathlib import Path
import pandas as pd

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