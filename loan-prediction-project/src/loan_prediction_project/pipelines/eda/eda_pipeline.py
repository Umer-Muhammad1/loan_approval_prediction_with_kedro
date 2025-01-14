from kedro.pipeline import Pipeline, node

from .eda_nodes import (generate_correlation_heatmap_by_order, plot_outliers_all_columns, plot_correlation_matrix,
plot_loan_acceptance_by_categorical_features, plot_distributions , remove_duplicates , plot_categorical_distributions,
plot_categorical_relations, plot_categorical_relations_grade, plot_histograms_kde)

def create_pipeline(**kwargs):
    preprocessing_pipeline= Pipeline(
        [
            node(
                func=generate_correlation_heatmap_by_order, 
                inputs=["raw_data", "params:output_path2"],
                outputs="correlation_matrix_by_order", 
                name="correlation"
                )
            ,
            node(
                func=remove_duplicates,
                inputs=["raw_data"],
                outputs="data_without_duplicates",
                name="removing_duplicates"
            ),
            node(
                func=plot_outliers_all_columns,
                inputs=["raw_data"],
                outputs="columns_outliers_plot",
                name="plot_outliers_all_columns_node",
            ),
            node(
                func= plot_correlation_matrix,
                inputs=["raw_data"],
                outputs= "correlation_matrix",
                name= "correlation_matrix",
                
            ),
            node(
                func= plot_loan_acceptance_by_categorical_features,
                inputs=["raw_data", 'params:categorical_features'],
                outputs= "categorical_features_by_loan_status",
                name= "categorical_features_by_loan_status",
                
            ),
            node(
                func=plot_distributions,
                inputs=["raw_data"],
                outputs="distribution_plot",
                name="plot_distributions_node"
            ),
            
            node(
                func=plot_categorical_distributions,
                inputs="data_without_duplicates",  # Input dataset (e.g., train DataFrame)
                outputs="categorical_features_distribution_plot",  # No outputs as we are showing plots
                name="plot_categorical_distributions_node"
            ),
            node(
                func=plot_categorical_relations,
                inputs=["data_without_duplicates", "params:categorical_features", "params:hue_feature"],  # Input dataset (e.g., train DataFrame)
                outputs="categorical_relation_plot",  # No outputs as we are showing plots
                name="plot_categorical_relations"
            ),
            node(
                func=plot_categorical_relations_grade,
                inputs=["data_without_duplicates", "params:categorical_features", "params:hue_feature"],  # Input dataset (e.g., train DataFrame)
                outputs="categorical_relation_plot_grade",  # No outputs as we are showing plots
                name="plot_categorical_relations_grade"
            ),
            
            
            node(
                func=plot_histograms_kde,
                inputs=["data_without_duplicates", "params:hist_columns", "params:hue_column"],
                outputs="kde_histogram",
                name="plot_histograms_node"
    )

        ]
    )
    return preprocessing_pipeline