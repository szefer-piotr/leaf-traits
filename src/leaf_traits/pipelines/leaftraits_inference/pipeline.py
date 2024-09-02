"""
The inference pipeline will take the figure and/or auxillary environmental data and predict the values of the sic target traits.
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_registered_model_pth, instantiate_the_model, load_the_model_state_dict
from src.leaf_traits.pipelines.data_processing.nodes import add_image_paths

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="get_registered_model",
            inputs=["params:registered_model_name"],
            outputs="model_pth",
            func=get_registered_model_pth,
        ),
        node(
            name="instantiate_the_model",
            inputs=["params:model_name", 
                    "params:feature_columns", 
                    "params:target_columns",
                    "params:device"],
            outputs="instantiated_model",
            func=instantiate_the_model,
        ),
        node(
            name="load_the_model_state_dict",
            inputs="model_pth",
            outputs="model",
            func=load_the_model_state_dict,
        ),
        node(
            name="add_train_data_file_paths",
            inputs=["test_raw","params:test_image_path"],
            outputs="test_with_image_paths",
            func=add_image_paths,
        ),
    ])