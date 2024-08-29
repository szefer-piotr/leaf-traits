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

from torch import nn
from torch.utils.data import DataLoader
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


def train_selected_model(
    model_name: str,
    saved_model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_columns: List,
    target_transformation: str,
    feature_columns: List,
    device: torch.device,
    epochs: int = 3,
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
        epochs=epochs,
        device=device)

    mlflow.pytorch.log_model(
        pytorch_model=model_results, 
        artifact_path="pytorch-model",
        registered_model_name="resnet_50_model")

    return model_results



# TODO: modify this function such that it uses model obtained from get_registered_model_pth funciton and node

def evaluate_model(
    test_data: pd.DataFrame,
    model: nn.Module,
    target_columns: List,
    feature_columns: List,
    target_transformation: str,
    test_batch_size: int,
    device: torch.device,
) -> float:
    
    model = model.to(device)

    test_transformations, val_transformations = create_augmentations(
        original_image_size= 512,
        image_size=288
        )

    test_dataset = LTDataset(
        test_data['file_path'].values,
        test_data[target_columns].to_numpy(),
        test_data[feature_columns].to_numpy(),
        target_transformation,
        test_transformations,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        drop_last=True,
        num_workers=1,
    )

    target_medians = test_data[target_columns].median(axis=0).values
    
    loss_fn = R2Loss(target_medians=target_medians, eps=1e-6)
    
    model.eval()
    
    val_loss = 0
    
    with torch.inference_mode():
        for batch, (X,y) in enumerate(test_dataloader):
            # print(f"Batch: {batch}")
            X['image'], X['feature'], y= X['image'].to(device), X['feature'].to(device), y.to(device)
            val_preds = model(X)
            loss = loss_fn(val_preds['targets'], y)
            val_loss += loss.item()
    
    val_loss = val_loss / len(test_dataloader)

    mlflow.log_metric("test_loss", val_loss)

    return val_loss