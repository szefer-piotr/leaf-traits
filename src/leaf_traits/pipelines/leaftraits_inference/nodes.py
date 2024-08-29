"""
This pipeline takes the test image and data and 
"""
import torch
import pandas as pd

from torch import nn
from typing import List
from mlflow.client import MlflowClient
from models.model_builders import (
     ViTModelConcat, 
     ViTModelAdd,
     ResNetModelAdd
)

def get_registered_model_pth(
    model_name: str ="resnet_50_model",
    tag: str = "inference",
):
    client = MlflowClient()
    model_info = client.get_model_version_by_alias(model_name, tag)
    
    artifacts = client.list_artifacts(
        run_id=model_info.run_id, 
        path='pytorch-model/data'
        )
    
    # model_pth = client.download_artifacts()

    print(f"[INFO] Returned object type {type(artifacts)}")
    
    for element in range(len(artifacts)):
        print(f"[INFO] Artifact {element} has a path {artifacts[element].path}")
        if artifacts[element].path.endswith("pth"):
            model_pth = client.download_artifacts(
                run_id=model_info.run_id,
                path=artifacts[element].path
            )
            print(f"[INFO] {model_pth} downloaded and is ready to use")

    print(f"[INFO] The type of the downloaded file is {type(model_pth)}.")
    print(model_pth)

    return model_pth

def instantiate_the_model(
    model_name: str,
    target_columns: List,
    feature_columns: List,
    device: torch.device,
):
    models_dict = {
        "vit_concat": ViTModelConcat, 
        "vit_add": ViTModelAdd,
        "resnet50_add": ResNetModelAdd}
    
    model = models_dict[model_name](
        n_features=len(feature_columns),
        n_targets=len(target_columns),
        device=device
    ).to(device)
    
    return model


def load_the_model_state_dict(
    model_instance: nn.Module,
    model_pth: str,
):
    """
    model_pth (str) is a path to the locally downloaded file.
    """

    state_dict = torch.load(model_pth)

    print(state_dict)
    
    # model = model_instance.load_state_dict()

    return state_dict

def predict_target_using_model(
    model_instance: nn.Module,
    test_data: pd.DataFrame,
    evaluation_function: callable,
):
    model_instance.eval()
    model_instance.pred
    with torch.inference_mode():
        model_instance(pd.DataFrame.to(device))