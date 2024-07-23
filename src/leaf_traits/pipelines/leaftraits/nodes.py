"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

import fsspec
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Callable, List
from sklearn.model_selection import train_test_split

from models.model_builders import ViTModelAdd, ViTModelConcat
from src.utils.save_model import save_model
from src.utils.datasets import LTDataset
from src.utils.data_setup import create_augmentations, create_dataloaders, visualize_transformations
from src.utils.engine import train_step, val_step, train_model



def download_data_from_github():
    
    data_path = Path("data/01_raw")
    image_path = data_path / "train_images"
    train_path = data_path / "train.csv"
    
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    
    else:
        print(f"Did not find {image_path} directory, downloading from GitHub...")
        fs = fsspec.filesystem("github", org="szefer-piotr", repo="ltdata")
        fs.get(fs.ls("train_images"), image_path.as_posix(), recursive=True)
        fs.get("train.csv", train_path.as_posix())
    
    return 1



def serialize_images(train_raw:pd.DataFrame, image_path:str):
    '''Reads an image file path from the raw data, then opens the corresponding image based on its id and writes it in a new column as bytes.

    Args:
        train_raw (pd.DataFrame): Raw train dataset.
        image_path (str): Path to the folder containing the images.

    Returns:
        Returns serialized train dataset as pickle format.
    '''
    
    train_raw['file_path'] = train_raw['id'].apply(lambda s: f'{image_path}/{s}.jpeg')
    train_raw['jpeg_bytes'] = train_raw['file_path'].apply(lambda fp: open(fp, 'rb').read())
    
    return train_raw



def train_validation_split(train_dataset: pd.DataFrame):
    no_of_rows = train_dataset.shape[0]
    train_idxs, val_idxs = train_test_split(
        range(0,no_of_rows),
        train_size=0.2,
        shuffle=True,
        random_state=42
        )
    return train_dataset.loc[train_idxs], train_dataset.loc[val_idxs]



# def train_selected_model(
#     train_df: pd.DataFrame,
#     val_df: pd.DataFrame,
#     target_columns: List,
#     feature_columns: List,
# ):
    
#     train_transformations, val_transformations = create_augmentations(
#         original_image_size= 512,
#         image_size=288
#         )
    
#     train_dataloader, val_dataloader = create_dataloaders(
#         train_df = train_df, 
#         val_df = val_df ,
#         target_columns = target_columns,
#         feature_columns = feature_columns,
#         train_transformations = train_transformations,
#         val_transformations = val_transformations,
#         train_batch_size = 8,
#         val_batch_size = 16,
#         )
    
