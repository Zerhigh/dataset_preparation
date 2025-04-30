from pathlib import Path
import pathlib
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt


base = Path("U:/master/dl2/combined_download")
file = base / 's2_ortho_download_data.csv'
file_ = base / 's2_ortho_download_data_try-without_vineyards.csv'


data_ = pd.read_csv(file_)
data = pd.read_csv(file)

fig, axs = plt.subplots(1, 2, sharex=True, figsize=(16, 8))

# Plot histograms
axs[0].hist(data_["low_corr"], bins=100, color="blue", edgecolor="black", alpha=0.7)
axs[0].set_ylabel("Frequency")
axs[0].set_title(f"Histogram of data without vineyards, num nodata in mask = {(data_['dist_0'] > 0).sum()} num nodata in general = {(data_['contains_nodata'] == True).sum()}")

axs[1].hist(data["low_corr"], bins=100, color="green", edgecolor="black", alpha=0.7)
axs[1].set_xlabel("Value")
axs[1].set_ylabel("Frequency")
axs[1].set_title(f"Histogram of data with vineyards, num nodata in mask = {(data['dist_0'] > 0).sum()} num nodata in general = {(data['contains_nodata'] == True).sum()}")

# Adjust layout
plt.tight_layout()
plt.show()