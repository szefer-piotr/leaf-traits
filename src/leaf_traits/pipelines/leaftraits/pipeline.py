"""
This is a boilerplate pipeline 'leaftraits'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import (
    train_selected_model,
    evaluate_model,
    # plot_loss_curves
)

from src.leaf_traits.pipelines.leaftraits_inference.nodes import get_registered_model_pth

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="train_model",
            inputs=[
                "params:model_name", 
                "params:save_model_name", 
                "train_df", 
                "val_df", 
                "params:target_columns",
                "params:target_transformation",
                "params:feature_columns", 
                "params:device",
                "params:epochs",
                "params:save_model_path",
            ],
            outputs="trained_model",
            func=train_selected_model,
        ),
        # node(
        #     name="plot_loss_curves",
        #     inputs="model_results",
        #     outputs="figure",
        #     func=plot_loss_curves,
        # ),
        node(
            name="get_registered_model",
            inputs=[],
            outputs="model_pth",
            func=get_registered_model_pth,
        ),
        node(
            name="evaluate_model",
            inputs=[
                "test_with_image_paths",
                "model_pth",
                "target_columns",
                "feature_columns",
                "target_transformation",
                "test_batch_size",
                "device",
           ],
           outputs="r2_fit",
           func=evaluate_model,
        ),
    ])
