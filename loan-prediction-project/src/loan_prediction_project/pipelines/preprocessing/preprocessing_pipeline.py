from kedro.pipeline import Pipeline, node

from .preprocessing_nodes import generate_correlation_matrix

def create_pipeline(**kwargs):
    preprocessing_pipeline= Pipeline(
        [
            node(
                func=generate_correlation_matrix, 
                inputs=["raw_data", "params:output_path"],
                outputs="correlation_matrix", 
                name="correlation"
                )
            ,
        ]
    )
    return preprocessing_pipeline