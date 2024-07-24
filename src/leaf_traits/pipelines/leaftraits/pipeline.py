"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    download_data_from_github, 
    serialize_images, 
    train_validation_split,
    train_selected_model
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="download_data",
            inputs=None,
            outputs="not important",
            func=download_data_from_github
        ),
        node(
            name="serialize_images",
            inputs=["train_raw", "params:image_path"],
            outputs="train_image_bytes",
            func=serialize_images,
        ),
        node(
            name="train_validation_split",
            inputs="train_image_bytes",
            outputs=["train_df", "val_df"],
            func=train_validation_split,
        ),
        node(
            name="train_model",
            inputs=[
                "params:model_name", 
                "params:save_model_name", 
                "train_df", 
                "val_df", 
                "params:target_columns",
                "params:feature_columns", 
                "params:device",
                "params:save_model_path"
            ],
            outputs="model_results",
            func=train_selected_model,
        )
    ])
