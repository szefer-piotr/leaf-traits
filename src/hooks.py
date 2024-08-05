from kedro.framework.hooks import hook_impl
from kedro.framework.context import KedroContext
from kedro.pipeline import Pipeline

from typing import Any

import mlflow
from mlflow.entities import RunStatus

class SimpleMLflow():
    def __init__(self):
        self.params = None
    
    @hook_impl
    def after_context_created(
        self,
        context: KedroContext,
    ) -> None:
        self.params = context.params

    @hook_impl
    def before_pipeline_run(
        self, run_params: dict[str, Any], pipeline: Pipeline, catalog
    ) -> None:        
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("testing_architectures")
        mlflow.start_run()
        mlflow.set_tag("test_pipeline", "example")
        if self.params:
            mlflow.log_params(self.params)

    @hook_impl
    def after_pipeline_run(
        self,
        run_params: dict[str, Any],
        run_result: dict[str, Any],
        pipeline: Pipeline,
        catalog,
    ) -> None:
        mlflow.end_run()

    def on_pipeline_error(
        self,
        error: Exception,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        catalog,
    ) -> None:
        mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))