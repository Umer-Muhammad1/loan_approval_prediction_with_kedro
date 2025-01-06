"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
#
from loan_prediction_project.pipelines.preprocessing import preprocessing_pipeline

def register_pipelines() -> dict:
    """Register the project's pipelines."""
    
    # Register the preprocessing pipeline
    data_processing_pipeline = preprocessing_pipeline.create_pipeline()

    # Placeholder for model training pipeline (currently empty)
    model_training_pipeline = Pipeline([])  # Empty pipeline for now

    # Combine pipelines if needed for default pipeline
    __default__ = data_processing_pipeline + model_training_pipeline
    
    return {
        "__default__": __default__,
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,
    }