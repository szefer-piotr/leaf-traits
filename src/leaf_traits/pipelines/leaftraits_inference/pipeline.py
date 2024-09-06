"""
The inference pipeline will take the figure and/or auxillary environmental data and predict the values of the sic target traits.
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_registered_model_pth, instantiate_the_model, load_model_state_dict, predict_target_using_model

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
            func=load_model_state_dict,
        ),
        # node(
        #     name="predict_target_using_model",
        #     inputs="model_request",
        #     outputs="model_response",
        #     func=predict_target_using_model
        # ),
    ])