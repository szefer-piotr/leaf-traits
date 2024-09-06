import pandas as pd
import torch
import matplotlib.pyplot as plt
import mlflow

from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List
from torchmetrics.regression import R2Score

from mlflow.client import MlflowClient

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
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_columns: List,
    target_transformation: str,
    feature_columns: List,
    device: torch.device,
    epochs: int = 3,
    registered_model_name: str = "resnet_50_model",
    artifact_path: str ="pytorch-model",
    registered_model_alias: str = "testing",
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
    
    model, model_results = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device)
    
    # Model gets logged in mlflow
    mlflow.pytorch.log_model(
        pytorch_model=model, 
        artifact_path=artifact_path,
        registered_model_name=registered_model_name)
    
    client = MlflowClient()
    mv = client.get_latest_versions(registered_model_name)
    # print(f"[INFO] Latest model version of the {registered_model_name} is {type(mv)}")
    
    client.set_registered_model_alias(
        registered_model_name, 
        registered_model_alias, 
        mv[0].version)

    # Train and validation losses are forwarded to the next node
    return model_results, model

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

def score_model(
    test_data: pd.DataFrame,
    model: nn.Module,
    target_columns: List,
    feature_columns: List,
    target_transformation: str,
    device: torch.device,
) -> float:
    
    model = model.to(device)

    test_transformations, _ = create_augmentations(
        original_image_size= 512,
        image_size=288
        )
    
    print(f"[INFO] Test transformations created...")

    test_dataset = LTDataset(
        test_data['file_path'].values,
        test_data[target_columns].to_numpy(),
        test_data[feature_columns].to_numpy(),
        target_transformation,
        test_transformations,
    )

    print(f"[INFO] Test dataset created...")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        drop_last=True,
        num_workers=1,
    )

    print(f"[INFO] Test dataloader created...")

    # Log metrics for the model
    r2score = R2Score(num_outputs=6).to(device)

    model.eval()
    
    val_score = 0
    
    with torch.inference_mode():
        for _, (X,y) in enumerate(test_dataloader):
            X['image'], X['feature'], y= X['image'].to(device), X['feature'].to(device), y.to(device)
            val_preds = model(X)
            val_score = r2score(val_preds['targets'], y)    
            val_score += val_score.item()
    
    val_score = val_score / len(test_dataloader)

    mlflow.log_metric("validation_score", val_score)

    return val_score