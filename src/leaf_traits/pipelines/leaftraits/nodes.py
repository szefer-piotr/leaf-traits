"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

import fsspec
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split



def download_data_from_github():
    '''
    Downloads the train tabular data and images from github repository.
    '''
    data_path = Path("data/01_raw")
    image_path = data_path / "train_images"
    train_path = data_path / "train.csv"
    
    if image_path.is_dir():
        print(f"On path: {image_path} directory exists.")
    
    else:
        print(f"Did not find {image_path} directory, downloading from GitHub...")
        fs = fsspec.filesystem("github", org="szefer-piotr", repo="ltdata")
        fs.get(fs.ls("train_images"), image_path.as_posix(), recursive=True)
        fs.get("train.csv", train_path.as_posix())
    
    return "Done"



def train_validation_split(train_raw, val_size, random_state):
    '''
    Uploads tabular train data
    '''
    train, val = train_test_split(
        train_raw, 
        test_size=val_size, 
        shuffle=True, 
        random_state=random_state
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    print(f"Train shape: {train.shape}\nValidation shape: {val.shape}")
    return train, val


def binarize_images():
    pass