"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import leaftraits


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="hello",
            inputs=None,
            outputs="not important",
            func=leaftraits
        )
    ])
