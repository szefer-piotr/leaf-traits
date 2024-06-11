"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import download_data_from_github


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="download_data",
            inputs=None,
            outputs="Dataset",
            func=download_data_from_github
        )
    ])
