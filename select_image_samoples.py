import pandas as pd
import rasterio
import pathlib
from pathlib import Path
import random
import numpy as np
import torch


def resample_torch(arr: np.ndarray, scale_factor: float | int, mode: str = 'bilinear') -> np.ndarray:
    """
    Args:
        arr: (channel x width x height) array will be resampled to (channel x scale_factor*width x scale_factor*height)
        scale_factor: float, 0.25 for HR -> LR
                             4 for HR -> LR

    Returns: ret_arr

    """
    antialis = True
    if mode == 'nearest':
        antialis = False

    ret_arr = torch.nn.functional.interpolate(
        torch.from_numpy(arr).unsqueeze(0),
        scale_factor=scale_factor,
        mode=mode,
        antialias=antialis
    ).squeeze().numpy()
    return ret_arr


def stretch(img):
    p_low, p_high = 2, 98
    stretched_bands = []

    for band in img:
        # Compute percentiles
        low = np.percentile(band, p_low)
        high = np.percentile(band, p_high)

        # Clip to percentile range
        clipped = np.clip(band, low, high)

        if high > low:
            # Min-max normalize to [0,1]
            norm = (clipped - low) / (high - low)
            # Scale to [0,255] and convert to uint8
            stretched = (norm * 255).astype(np.uint8)
        else:
            stretched = np.zeros_like(band, dtype=np.uint8)

        stretched_bands.append(stretched)

    return np.stack(stretched_bands)

data_path = Path('/data/USERS/shollend/metadata/stratification_tables/filtered/test.csv')
s2_path = Path('/data/USERS/shollend/combined_download/output/lr_s2')
to_path = Path('/data/USERS/shollend/image_samples')

sr_models = {
    "sr4rs": "/data/USERS/shollend/sentinel2/sr_inference/sr4rs",
    "swin2mose": "/data/USERS/shollend/sentinel2/sr_inference/swin2_mose/swin2_mose_experiment",
    "sen2sr_rgbn": "/data/USERS/shollend/sentinel2/sr_inference/SEN2SR_RGBN",
    "bilinear": "/data/USERS/shollend/sentinel2/sr_inference/bilinear",
    "deepsent": "/data/USERS/shollend/sentinel2/sr_inference/deepsent/predicted_merged",
    "ldrs2": "/data/USERS/shollend/sentinel2/sr_inference/diffusion_updated_simon/sr_ims_Austria",
    "evoland": "/data/USERS/shollend/sentinel2/sr_inference/evoland/experiment_resolution",
    "sen2sr_lite": "/data/USERS/shollend/sentinel2/sr_inference/SEN2SRLite_RGBN",
    "ortho": "/data/USERS/shollend/combined_download/output/hr_orthofoto",
    "nn": None,
    "bicubic": None,}

data = pd.read_csv(data_path)
sampeld = pd.concat([df.sample(n=3, random_state=32) for i, df in data.groupby(data['assigned_class'])], axis=0)
print(sampeld.head())

for i, row in sampeld.iterrows():
    img_id = f"{row['id']:05d}"
    save_path = to_path / img_id
    save_path.mkdir(exist_ok=True)

    # get s2
    s2_id = f"S2_{img_id}.tif"
    s2_img_path = s2_path / s2_id

    with rasterio.open(s2_img_path, 'r') as s2_src:
        # get rgb
        s2_img = s2_src.read([4, 3, 2])
        s2_img = stretch(s2_img)

        new_profile = s2_src.profile.copy()
        new_profile['count'] = 3
        new_profile['dtype'] = 'uint8'
        new_profile['nodata'] = None

        with rasterio.open(save_path / s2_id, 'w', **new_profile) as s2_dst:
            s2_dst.write(s2_img)

    # get sr iamges
    for model, sr_path in sr_models.items():
        if sr_path is not None:
            sr_path = Path(sr_path)
        sr_profile = None
        sr_id = None
        sr_save_id = None
        sr_img = None
        sr_bands = None

        if model in ("nn", "bicubic"):
            if model == "nn":
                sr_img = resample_torch(s2_img, scale_factor=4, mode="nearest")
            elif model == "bicubic":
                sr_img = resample_torch(s2_img, scale_factor=4, mode="bicubic")

            b, h, w = sr_img.shape
            sr_save_id = f"SR_{model}_{img_id}.tif"

            # load bilinear trafo
            with rasterio.open(save_path / f"SR_bilinear_{img_id}.tif", 'r') as bil_src:

                sr_profile = {
                    "dtype": sr_img.dtype,
                    "height": h,
                    "width": w,
                    "count": 3,
                    "transform": bil_src.transform,
                    "crs": bil_src.crs,
                    'driver': 'GTiff',
                    'nodata': None,
                }

                with rasterio.open(save_path / sr_save_id, 'w', **sr_profile) as sr_dst:
                    sr_dst.write(sr_img)

            continue

        elif model == "ortho":
            sr_id = f"HR_ortho_{img_id}.tif"
            sr_bands = [1 , 2, 3]
            sr_save_id = f"HR_ortho_{img_id}.tif"
        elif model == "bilinear":
            sr_id = f"S2_{img_id}.tif"
            sr_save_id = f"SR_bilinear_{img_id}.tif"
            sr_bands = [4, 3, 2]
        else:
            sr_id = f"S2_{img_id}.tif"
            sr_save_id = f"SR_{model}_{img_id}.tif"
            sr_bands = [1, 2, 3]

        with rasterio.open(sr_path / sr_id, 'r') as sr_src:
            print(model, sr_bands)
            sr_img = sr_src.read(sr_bands)
            sr_img = stretch(sr_img)

            if sr_profile is None:
                sr_profile = sr_src.profile.copy()

            sr_profile['count'] = 3
            sr_profile['dtype'] = 'uint8'
            sr_profile['nodata'] = None

            with rasterio.open(save_path / sr_save_id, 'w', **sr_profile) as sr_dst:
                sr_dst.write(sr_img)

    # gt masks

    # pred masks

    break




