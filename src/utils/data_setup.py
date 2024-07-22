from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
from pathlib import Path
import albumentations as A

from typing import Tuple, List
from src.utils.datasets import LTDataset
import pandas as pd
from albumentations.pytorch import ToTensorV2

# Add augmentations
def create_augmentations(
    original_image_size: int,
    image_size: int,
) -> Tuple[A.Compose, A.Compose]:
    
    """Create basic augmentations.

    Creates image augmentations using the augmentations library.
    
    Args:
        original_image_size: integer with original image size value.
        image_size: integer determining image size that is going to be fed to the model

    Returns:
        A tuple containing transformation obbjects for the training and validation images.
    """
    
    TRAIN_TRANSFORMS = A.Compose([
        A.RandomSizedCrop(
            [int(0.85*original_image_size), original_image_size],
            image_size, image_size, w2h_ratio=1.0, p=1.0
        ),
        A.HorizontalFlip(p=0.50),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.50),
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
        ToTensorV2(),
    ])

    VAL_TRANSFORMS = A.Compose([
            A.Resize(image_size,image_size),
            ToTensorV2(),
        ])
    
    return TRAIN_TRANSFORMS, VAL_TRANSFORMS


def create_dataloaders(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame,
    target_columns: List,
    feature_columns: List,
    train_transformations: A.Compose,
    val_transformations: A.Compose,
    train_batch_size: int,
    val_batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = LTDataset(
        train_df['file_path'].values,
        train_df[target_columns].to_numpy(),
        train_df[feature_columns].to_numpy(),
        train_transformations,
    )

    val_dataset = LTDataset(
        val_df['file_path'].values,
        val_df[target_columns].to_numpy(),
        val_df[feature_columns].to_numpy(),
        val_transformations,
    )

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            drop_last=True,
            num_workers=1,
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        drop_last=True,
        num_workers=1
    )

    return train_dataloader, val_dataloader
