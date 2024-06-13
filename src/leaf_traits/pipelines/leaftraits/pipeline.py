"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    download_data_from_github,
    train_validation_split,
    serialize_images
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
            inputs=["train_raw", "params:N_VAL_SAMPLES", "params:SEED"],
            outputs=["train","val"],
            func=train_validation_split,
        )
    ])
