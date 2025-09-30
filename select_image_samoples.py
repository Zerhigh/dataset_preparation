import shutil
from asyncio import wait_for

import pandas as pd
import rasterio
from pathlib import Path
import numpy as np
import torch
import tqdm
from PIL import Image


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
gt_path = Path('/data/USERS/shollend/combined_download/output/hr_mask')
pred_path = Path('/data/USERS/shollend/inferred_buildings')
tile_coords = [(0, 0), (0, 256), (256, 0), (256, 256)]

to_path = Path('/data/USERS/shollend/image_samples')
to_path_png = to_path / 'png'
to_path_tif = to_path / 'tif'

to_path_png.mkdir(exist_ok=True)
to_path_tif.mkdir(exist_ok=True)

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


for i, row in tqdm.tqdm(sampeld.iterrows()):
    img_id = f"{row['id']:05d}"
    save_path_png = to_path_png / img_id
    save_path_tif = to_path_tif / img_id

    save_path_png.mkdir(exist_ok=True)
    save_path_tif.mkdir(exist_ok=True)

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

        with rasterio.open(save_path_tif / s2_id, 'w', **new_profile) as s2_dst:
            s2_dst.write(s2_img)

        # save as pnh
        png_s2_img = np.transpose(s2_img, (1, 2, 0))
        s2_id_xx = s2_id.replace('tif', 'png')
        Image.fromarray(png_s2_img).save(save_path_png / s2_id_xx)

        s2tile_coords = [(0, 0), (0, 32), (32, 0), (32, 32)]
        for i, (top, left) in enumerate(s2tile_coords):
            s2_tile = png_s2_img[top:top + 32, left:left + 32, :]
            Image.fromarray(s2_tile).save(save_path_png / f'{i}_{s2_id_xx}')

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
            with rasterio.open(save_path_tif / f"SR_bilinear_{img_id}.tif", 'r') as bil_src:

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

                with rasterio.open(save_path_tif / sr_save_id, 'w', **sr_profile) as sr_dst:
                    sr_dst.write(sr_img)

            # save as pnh
            png_sr_img = np.transpose(sr_img, (1, 2, 0))
            sr_save_id_xx = sr_save_id.replace('tif', 'png')
            Image.fromarray(png_sr_img).save(save_path_png / sr_save_id_xx)
            for i, (top, left) in enumerate(tile_coords):
                mask_tile = png_sr_img[top:top + 256, left:left + 256, :]
                Image.fromarray(mask_tile).save(save_path_png / f'{i}_{sr_save_id_xx}')

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
            sr_img = sr_src.read(sr_bands)
            sr_img = stretch(sr_img)

            if sr_profile is None:
                sr_profile = sr_src.profile.copy()

            sr_profile['count'] = 3
            sr_profile['dtype'] = 'uint8'
            sr_profile['nodata'] = None

            with rasterio.open(save_path_tif / sr_save_id, 'w', **sr_profile) as sr_dst:
                sr_dst.write(sr_img)

        # save as pnh
        png_sr_img = np.transpose(sr_img, (1, 2, 0))
        sr_save_id_xx = sr_save_id.replace('tif', 'png')
        Image.fromarray(png_sr_img).save(save_path_png / sr_save_id_xx)
        for i, (top, left) in enumerate(tile_coords):
            mask_tile = png_sr_img[top:top + 256, left:left + 256, :]
            Image.fromarray(mask_tile).save(save_path_png / f'{i}_{sr_save_id_xx}')

    # gt masks
    gt_id = f"HR_mask_{img_id}.tif"
    gt_img_path = gt_path / gt_id
    with rasterio.open(gt_img_path, 'r') as gt_src:
        # get rgb
        gt_img = gt_src.read()
        mask = gt_src.read(1).astype(np.uint8)
        mask = (mask == 41).astype(np.uint8) * 255

        gt_profile = gt_src.profile.copy()
        gt_profile['count'] = 1
        gt_profile['dtype'] = 'uint8'
        gt_profile['nodata'] = None

        with rasterio.open(save_path_tif / gt_id, 'w', **gt_profile) as gt_dst:
            gt_dst.write(mask, 1)

        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        gt_id_xx = gt_id.replace('tif', 'png')
        Image.fromarray(mask_rgb).save(save_path_png / gt_id_xx)

        # also save as clipped images
        for i, (top, left) in enumerate(tile_coords):
            mask_tile = mask_rgb[top:top + 256, left:left + 256, :]
            Image.fromarray(mask_tile).save(save_path_png / f'{i}_{gt_id_xx}')

    # pred masks
    pred_id = f"S2_{img_id}.tif"
    for model, _ in sr_models.items():
        pred_img_path = pred_path / model / "predicted" / pred_id
        colored_img_path = pred_path / model / "colored" / pred_id

        to_pred_tif = save_path_tif / f'PRED_SR_{model}_predicted_{img_id}.tif'
        to_pred_png = save_path_png / f'PRED_SR_{model}_predicted_{img_id}.png'

        to_col_tif = save_path_tif / f'PRED_SR_{model}_colored_{img_id}.tif'
        to_col_png = save_path_png / f'PRED_SR_{model}_colored_{img_id}.png'

        # just copy tif
        shutil.copy(src=pred_img_path, dst=to_pred_tif)
        shutil.copy(src=colored_img_path, dst=to_col_tif)

        # do pngs
        with rasterio.open(pred_img_path, 'r') as pred_img_src:
            pred_mask = pred_img_src.read(1).astype(np.uint8) * 255

            pred_mask_rgb = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)
            Image.fromarray(pred_mask_rgb).save(to_pred_png)

        with rasterio.open(colored_img_path, 'r') as col_img_src:
            col_mask = col_img_src.read(1).astype(np.uint8)

            col_mask_rgb = np.stack([col_mask, col_mask, col_mask], axis=-1)
            # np_colored[tp] = 1
            # np_colored[fp] = 2
            # np_colored[fn] = 3
            # np_colored[tn] = 0

            col_mask_rgb[col_mask == 1, 1] = 255 #green
            col_mask_rgb[col_mask == 2, 0] = 255 #red
            col_mask_rgb[col_mask == 3, 2] = 255 # blue

            Image.fromarray(col_mask_rgb).save(to_col_png)




