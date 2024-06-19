"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

import fsspec
import pandas as pd
import imageio.v3 as  imageio
import albumentations as A

from pathlib import Path
from sklearn.model_selection import train_test_split
from kedro_datasets import pickle
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


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



# def outlier_handler(train_raw: pd.DataFrame) -> Tuple[train_filtered]:
#     '''Function that handles outliers. 
#     There are many values in the datasets, that are extremely large or negative. 
#     These situations should not occur, therefore negative values are removed, and large values above xxx quantile
#     are also removed.

#     Args:
#         placeholder

#     Returns:
#         Filtered DataFrame.
#     '''

#     return mask



def train_validation_split(train_serialized, val_size:int, random_state:int):
    '''Reads serialized pickle data and splits it into train and validation sests.

    Args:
        train_serialized: pickle.PickleDataset

    '''

    # print(f"Reading pickle file from {train_serialized_path}")
    # data = pd.read_pickle(train_serialized_path)
    print(type(train_serialized))

    # train0 = train_serialized.load()

    train, val = train_test_split(
        train_serialized, 
        test_size=val_size, 
        shuffle=True, 
        random_state=random_state
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    print(f"Train shape: {train.shape}\nValidation shape: {val.shape}")
    return train, val

def get_features(train: pd.DataFrame, FEATURES: List) -> pd.DataFrame:
    return train[FEATURES]

def get_targets(train: pd.DataFrame, TARGETS: List) -> pd.DataFrame:
    return train[TARGETS]

def get_images(train: pd.DataFrame, IMAGE_COLUMN: List = ['jpeg_bytes']) -> pd.DataFrame:
    return train[IMAGE_COLUMN]

def create_dataloader(
        data_mask: pd.DataFrame, 
        y_data_mask: pd.DataFrame,
        data_feature_mask: pd.DataFrame,
        transformations: A.Compose = None,
        batch_size: int = 64,
        shuffle: bool = True) -> DataLoader:
    
    class LTDataset(Dataset):
        def __init__(self, images, targets, features, transforms=None):
            self.images = images
            self.targets = targets
            self.features = features
            self.transforms = transforms
        
        def __len__(self):
            len(self.images)

        def __getitem__(self, index):
            X_sample = {
                'image': self.transforms(
                    image=imageio.imread(self.images[index])
                    )['image'],
                'feature': self.features[index]    
                }
            
            y_sample = self.targets[index]
            
            return X_sample, y_sample
        
    dataset = LTDataset(
        data_mask,
        y_data_mask, # pass target names
        data_feature_mask, # pass feature names
        transformations
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )

    return dataloader
