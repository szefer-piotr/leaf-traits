"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from src.leaf_traits.pipelines.data_processing.pipeline import create_pipeline as data_processing_pipeline
from src.leaf_traits.pipelines.leaftraits.pipeline import create_pipeline as model_training_pipeline
from src.leaf_traits.pipelines.leaftraits_inference.pipeline import create_pipeline as inference_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    pipelines["__default__"] = data_processing_pipeline()
    pipelines['model_training_pipeline'] = model_training_pipeline()
    pipelines['inference_pipeline'] = inference_pipeline()
    return pipelines
