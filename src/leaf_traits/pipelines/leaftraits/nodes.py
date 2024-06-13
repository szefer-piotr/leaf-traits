"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

import fsspec
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from kedro_datasets import pickle

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



def train_validation_split(train_serialized, val_size:int, random_state:int):
    '''Reads serialized pickle data and splits it into train and validation sests.

    Args:
        train_serialized: pickle.PickleDataset

    '''

    # print(f"Reading pickle file from {train_serialized_path}")
    # data = pd.read_pickle(train_serialized_path)
    print(type(train_serialized))

    train0 = train_serialized.load()

    train, val = train_test_split(
        train0, 
        test_size=val_size, 
        shuffle=True, 
        random_state=random_state
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    print(f"Train shape: {train.shape}\nValidation shape: {val.shape}")
    return train, val


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
    
    return train_raw.to_pickle