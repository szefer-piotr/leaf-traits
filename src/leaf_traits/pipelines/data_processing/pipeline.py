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
    get_mask_for_outliers,
#     normalize_features
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
          name="download_example_data_from_github_repo",
          inputs="params:data_repository",
          outputs="image_and_feature_paths",
          func=download_data_from_github,
        ),
        node(
          name="add_train_image_paths_columns",
          inputs=["train_raw", "params:train_image_path"],
          outputs="train_with_image_paths",
          func=add_image_paths,
        ),
        node(
          name="add_test_image_paths_columns",
          inputs= ["test_raw", "params:test_image_path"],
          outputs="test_with_image_paths",
          func=add_image_paths,
        ),
     #    [TODO]
     #     node(
     #      name="noramlize_features",
     #      inputs=[],
     #      outputs=[],
     #      func=normalize_features,
     #    ),
        node(
          name="split_train_data_into_train_val",
          inputs=["train_with_image_paths", "params:train_size"],
          outputs=["train_df", "val_df"],
          func=train_validation_split,
        ),
    ])
