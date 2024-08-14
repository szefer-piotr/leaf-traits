"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import fsspec
import os

# def download_data_from_github(
#     data_path: str = "data/01_raw",
#     train_image_folder: str = "train_images",
#     train_features: str = "train.csv",
#     test_image_folder: str = "test_images",
#     test_features: str = "test.csv",
#     org: str = "szefer-piotr",
#     repo: str = "ltdata"
# ):

def download_data_from_github(
    data_repository: Dict,
):
    '''Downlods train and test image files path from the Github repository.

    Args:
        data_path (Path) defaults to "data/01_raw".
        train_image_folder (str) defaults to "train_images",
        train_features: str = "train.csv",
        test_image_folder: str = "test_images",
        test_features: str = "test.csv",
        org: str = "szefer-piotr",
        repo: str = "ltdata"

    Returns:
        Additionally to downloading the data it returns a tuple with paths to train images, train feature and test images and test features.
    '''
    data_path = Path(data_repository["data_path"])
    train_image_folder = data_repository["train_image_folder"]
    train_features = data_repository["train_features"]
    test_image_folder = data_repository["test_image_folder"]
    test_features = data_repository["test_features"]
    org = data_repository["org"]
    repo = data_repository["repo"]

    print(data_path)

    # data_path = Path("data/01_raw")
    train_image_path = data_path / train_image_folder
    train_features_path = data_path / train_features
    test_image_path = data_path / test_image_folder
    test_features_path = data_path / test_features

    required_files = ['train_images', 'test_images', 'train.csv', 'test.csv']
    directories_on_path = os.listdir(data_path)
    
    # print(f"[INFO] Files in the {data_path} path: {checklist}")

    if all([element in directories_on_path for element in required_files]):
        print(f"[INFO] Test and training data on {data_path} exist.")
    
    else:
        print(f"Did not find data in {data_path} directory, downloading from GitHub...")
        
        fs = fsspec.filesystem("github", org=org, repo=repo)
        
        fs.get(fs.ls("train_images"), train_image_path.as_posix(), recursive=True)
        fs.get("train.csv", train_features_path.as_posix())
        
        fs.get(fs.ls("test_images"), test_image_path.as_posix(), recursive=True)
        fs.get("test.csv", test_features_path.as_posix())

    return (train_image_path, train_features_path, test_image_path, test_features_path)



def add_image_paths(train_raw:pd.DataFrame, image_path:str):
    '''Reads an image file path from the raw data, then opens the corresponding image based on its id and writes it in a new column as bytes.

    Args:
        train_raw (pd.DataFrame): Raw train dataset.
        image_path (str): Path to the folder containing the images.

    Returns:
        Returns serialized train dataset as pickle format.
    '''
    
    train_raw['file_path'] = train_raw['id'].apply(lambda s: f'{image_path}/{s}.jpeg')
    # train_raw['jpeg_bytes'] = train_raw['file_path'].apply(lambda fp: open(fp, 'rb').read())
    
    return train_raw



def train_validation_split(
        train_dataset: pd.DataFrame, 
        train_size: float = 0.2,
    ):
    no_of_rows = train_dataset.shape[0]
    train_idxs, val_idxs = train_test_split(
        range(0,no_of_rows),
        train_size=train_size,
        shuffle=True,
        random_state=42
        )
    return train_dataset.loc[train_idxs], train_dataset.loc[val_idxs]






#[TODO] -----------





def calculate_target_outlier_values_based_on_quantiles(
    dataset: pd.DataFrame,
    target_columns: List,
    quantiles: Tuple[float, float] = (0.001, 0.999)
):
    # Minimum/Maximum Based On Train 0.1% and 99.9%
    v_min = dataset[target_columns].quantile(quantiles[0])
    v_max = dataset[target_columns].quantile(quantiles[1])

    return (v_min, v_max)

# Mask to exclude values outside of 0.1% - 99.9% range
def get_mask_for_outliers(
        df: pd.DataFrame,
        target_columns: List,
        outlier_quantile_values: Tuple,
    ) -> pd.DataFrame:
    
    lower = []
    higher = []
    
    mask = np.empty(shape=df[target_columns].shape, dtype=bool)
    
    # Fill mask based on minimum/maximum values of sample submission
    
    outlier_table = enumerate(
        zip(target_columns, 
            outlier_quantile_values[0],
            outlier_quantile_values[1]
            )
        )

    for idx, (t, v_min, v_max) in outlier_table:
        labels = df[t].values
        mask[:,idx] = ((labels > v_min) & (labels < v_max))
    
    return mask.min(axis=1)

    # Masks
    
    # CONFIG.MASK_TRAIN = get_mask(train)
    # Get mask for the train dataset and for the validation dataset
    
    # CONFIG.MASK_VAL = get_mask(val)
    
    # Masked DataFrames
    # train_mask = train[CONFIG.MASK_TRAIN].reset_index(drop=True)
    # val_mask = val[CONFIG.MASK_VAL].reset_index(drop=True)
    
    # Add Number Of Steps
    # CONFIG.N_TRAIN_SAMPLES = len(train_mask)
    # CONFIG.N_VAL_SAMPLES = len(val_mask)
    # CONFIG.N_STEPS_PER_EPOCH = (CONFIG.N_TRAIN_SAMPLES // CONFIG.BATCH_SIZE)
    # CONFIG.N_VAL_STEPS_PER_EPOCH = math.ceil(CONFIG.N_VAL_SAMPLES / CONFIG.BATCH_SIZE_VAL)
    # CONFIG.N_STEPS = CONFIG.N_STEPS_PER_EPOCH * CONFIG.N_EPOCHS + 1

    # for m, subset in zip([CONFIG.MASK_TRAIN, CONFIG.MASK_VAL], ['train', 'val']):
    #     print(f'===== {subset} shape: {m.shape} =====')
    #     print(f'{subset} \t| # Masked Samples: {(1-m.mean())*CONFIG.N_TRAIN_SAMPLES:.0f}')
    #     print(f'{subset} \t| % Masked Samples: {100-m.mean()*100:.3f}%')