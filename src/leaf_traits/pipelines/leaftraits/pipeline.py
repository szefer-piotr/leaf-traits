"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import download_data_from_github, serialize_images


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
    ])
