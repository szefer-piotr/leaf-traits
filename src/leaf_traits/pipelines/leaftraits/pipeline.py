"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    train_selected_model,
    score_model,
    plot_loss_curves
)

# from src.leaf_traits.pipelines.leaftraits_inference.nodes import (
#     get_registered_model_pth, 
#     instantiate_the_model,
#     load_model_state_dict
# )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="train_model",
            inputs=[
                "params:model_name", 
                "train_df", 
                "val_df", 
                "params:target_columns",
                "params:target_transformation",
                "params:feature_columns", 
                "params:device",
                "params:epochs",
                "params:registered_model_name", 
                "params:artifact_path",
                "params:registered_model_alias"
            ],
            outputs=["model_results", "model"],
            func=train_selected_model,
        ),
        node(
            name="plot_loss_curves",
            inputs="model_results",
            outputs="figure",
            func=plot_loss_curves,
        ),
        node(
            name="score_the_model",
            inputs=[
                "val_df",
                "model",
                "params:target_columns",
                "params:feature_columns",
                "params:target_transformation",
                "params:device"],
            outputs="model_score",
            func=score_model,
            tags="score",
        ),
    ])
