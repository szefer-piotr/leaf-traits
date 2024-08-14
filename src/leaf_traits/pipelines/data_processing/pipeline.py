"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    download_data_from_github,
    add_image_paths,
    train_validation_split,
    calculate_target_outlier_values_based_on_quantiles,
    get_mask_for_outliers
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(name="download_example_data_from_github_repo",
             inputs="params:data_repository",
             outputs="image_and_feature_paths",
             func=download_data_from_github),
    ])
