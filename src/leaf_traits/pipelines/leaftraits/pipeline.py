"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    download_data_from_github,
    train_validation_split,
    serialize_images,
    get_features,
    get_images,
    get_targets,
    create_dataloader
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="download_data",
            inputs=None,
            outputs="message",
            func=download_data_from_github,
        ),
        node(
            name="serialize_images",
            inputs=["train_raw","params:image_path"],
            outputs="train_serialized",
            func=serialize_images,
        ),
        node(
            name="train_val_split",
            inputs=["train_serialized", "params:N_VAL_SAMPLES", "params:SEED"],
            outputs=["train","val"],
            func=train_validation_split,
        ),
        node(
            name="get_images",
            inputs="train",
            outputs="train_images",
            func=get_images,
        ),
        node(
            name="get_tragets",
            inputs=["train", "params:TARGET_COLUMNS"],
            outputs="train_targets",
            func=get_targets,
        ),
        node(
            name="get_features", 
            inputs=["train","params:FEATURE_COLUMNS"],
            outputs="train_features",
            func=get_features,
        ),
        node(
            name="create_train_dataloader",
            inputs=["train_images", 
                    "train_targets", 
                    "train_features", 
                    "params:TRANSFORMATIONS", 
                    "params:BATCH_SIZE", 
                    "params:SHUFFLE"],
            outputs="train_dataloader",
            func=create_dataloader
        ),    
    ])
