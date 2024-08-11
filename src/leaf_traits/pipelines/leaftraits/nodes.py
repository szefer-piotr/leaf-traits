"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

import fsspec
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from pathlib import Path
from typing import Dict, Any, Callable, List
from sklearn.model_selection import train_test_split

from models.model_builders import (
    ViTModelAdd, 
    ViTModelConcat,
    ResNetModelAdd,
)
from src.utils.save_model import save_model
from src.utils.loss_function import R2Loss
from src.utils.datasets import LTDataset
from src.utils.data_setup import create_augmentations, create_dataloaders, visualize_transformations
from src.utils.engine import train_step, val_step, train_model


# from mlflow.models import infer_signature

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
    # train_raw['jpeg_bytes'] = train_raw['file_path'].apply(lambda fp: open(fp, 'rb').read())
    
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



def train_selected_model(
    model_name: str,
    saved_model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_columns: List,
    target_transformation: str,
    feature_columns: List,
    device: torch.device,
    save_model_path: str = "/models/",
):
    models_dict = {
        "vit_concat": ViTModelConcat, 
        "vit_add": ViTModelAdd,
        "resnet50_add": ResNetModelAdd}

    train_transformations, val_transformations = create_augmentations(
        original_image_size= 512,
        image_size=288
        )
    
    train_dataloader, val_dataloader = create_dataloaders(
        train_df = train_df, 
        val_df = val_df ,
        target_columns = target_columns,
        feature_columns = feature_columns,
        train_transformations = train_transformations,
        val_transformations = val_transformations,
        target_transformation = target_transformation,
        train_batch_size = 8,
        val_batch_size = 16,
        )
    
    model = models_dict[model_name](
        n_features=len(feature_columns),
        n_targets=len(target_columns),
        device=device
    ).to(device)

    target_medians=train_df[target_columns].median(axis=0).values
    
    loss_fn = R2Loss(target_medians=target_medians, eps=1e-6)

    optimizer = torch.optim.AdamW(
    params = model.parameters(),
    lr = 3e-4, #CONFIG.LR_MAX
    weight_decay = 0.01, #CONFIG.WEIGHT_DECAY
    )
    
    model_results = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=3,
        device=device)

    # save_model(model, target_dir=save_model_path, model_name=saved_model_name)
    mlflow.pytorch.log_model(model, artifact_path="model")

    return model_results

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    val_loss = results['validation_loss']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot loss
    # plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    mlflow.log_figure(fig, "train_val_loss.png")

    return fig