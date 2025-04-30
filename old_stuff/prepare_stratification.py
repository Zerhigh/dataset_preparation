# execute this script before a train test split to modiFy the stratiffication to remove unneccessary image containing non classes

import pathlib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


BASE = Path("C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/")
meta = pd.read_csv(BASE / "final_taco_metadata.csv")

meta['has_buildings'] = meta['dist_41'].apply(lambda x: False if x == 0 else True)
meta_buildings = meta[meta['has_buildings'] == True]
meta_no_buildings = meta[meta['has_buildings'] == False]

percent = 0.05
num_rows = int((percent/(1-percent)) * len(meta_buildings))  # num of sampels to have 5% of total sample number non-buildings

sampled_no_buildings = meta_no_buildings.sample(n=num_rows, random_state=32)

stratified_data = pd.concat([meta_buildings, sampled_no_buildings], axis=0)
stratified_data["strat_class"] = stratified_data["dist_41"].apply(assign_class)
stratified_data['test_train_val'] = None

plot = False
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot first histogram (dist_41)
    axes[0].hist(stratified_data['dist_41'], bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram of dist_41')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].hist(stratified_data['strat_class'], bins=5, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram of dist_41 for Each strat_class')
    axes[1].legend(title='strat_class')  # Add legend to differentiate the classes
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


train_ratio = 0.7  # 70% for training
val_ratio = 0.15   # 15% for validation
test_ratio = 0.15  # 15% for testing


train, temp = train_test_split(
    stratified_data, test_size=(1 - train_ratio), stratify=stratified_data["strat_class"], random_state=39
)

# Then, split temp into validation and test sets
val, test = train_test_split(
    temp, test_size=(test_ratio / (test_ratio + val_ratio)),
    stratify=temp["strat_class"], random_state=39
)

train["test_train_val"] = 0
test["test_train_val"] = 1
val["test_train_val"] = 2

stratified_table = pd.concat([train, test, val], axis=0)
stratified_table.to_csv('tables/stratified_metadata.csv')

pass