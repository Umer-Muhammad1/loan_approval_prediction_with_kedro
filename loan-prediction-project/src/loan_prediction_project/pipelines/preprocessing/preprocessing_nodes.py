import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



from datetime import datetime


def generate_correlation_matrix(raw_data: pd.DataFrame , output_path) -> pd.DataFrame:
    correlation_matrix = raw_data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title("Correlation Matrix")
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return correlation_matrix  # Ensure the correlation matrix is returned
