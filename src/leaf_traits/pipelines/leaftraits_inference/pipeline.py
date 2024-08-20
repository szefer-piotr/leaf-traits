"""
The inference pipeline will take the figure and/or auxillary environmental data and predict the values of the sic target traits.
"""

from kedro.pipeline import Pipeline, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])
