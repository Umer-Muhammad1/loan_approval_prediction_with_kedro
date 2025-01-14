import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gensim
from gensim.models import Word2Vec

from datetime import datetime


def generate_correlation_heatmap_by_order(data: pd.DataFrame, output_path: str) -> None:
    """
    Generate a heatmap showing the correlation of numeric features with 'loan_status'
    and save it as an image.

    Args:
        data: Input dataset containing numeric columns.
        output_path: Path to save the generated heatmap image.
    """
    # Compute correlation with 'loan_status'
    correlation_with_loan_status = data.corr(numeric_only=True)['loan_status'].sort_values(ascending=False)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the heatmap
    sns.heatmap(
        pd.DataFrame(correlation_with_loan_status),
        annot=True,
        cmap='coolwarm',
        fmt=".3f",
        ax=ax
    )

    # Set title
    ax.set_title('Correlation with Loan Status', fontsize=14)

    # Adjust layout
    fig.tight_layout()

    # Capture the figure
    img = plt.gcf()

    # Return the figure object
    return img
   
def plot_correlation_matrix(data):
    """
    Generate a heatmap for the entire correlation matrix of numeric features.

    Args:
        data: Input dataset containing numeric columns.

    Returns:
        img: The generated figure object.
    """
    # Compute the correlation matrix
    correlation_matrix = data.corr(numeric_only=True)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Create the heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        ax=ax,  # Use the axis created in the figure
        cbar_kws={"shrink": 0.8}  # Shrink color bar for better layout
    )

    # Set the title
    ax.set_title('Correlation Matrix Heatmap', fontsize=14)

    # Adjust layout
    fig.tight_layout()

    # Get the current figure object
    img = plt.gcf()

    # Return the figure object
    return img

    

def plot_outliers_all_columns(data):
 
   
    # Initialize the figure
    numerical_columns = data.select_dtypes(include=["number"]).columns
    n_features = len(numerical_columns)
    n_cols = 3  # Number of columns in the grid
    n_rows = 3  # Calculate rows needed

    # Create the grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, n_rows * 5))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Generate custom colors for each boxplot
    custom_colors = sns.color_palette("husl", len(numerical_columns))

    # Generate boxplots for each numerical column
    for i, col in enumerate(numerical_columns):
        sns.boxplot(
            x=data[col], 
            ax=axes[i], 
            color=custom_colors[i]  # Apply unique color to each boxplot
        )
        axes[i].set_title(f'Outliers for {col}', fontsize=12)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        # Create a figure with subplots
    
        
    
 # Set titles and labels
    
  
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img
            
        

def plot_loan_acceptance_by_categorical_features(data, categorical_features):
    """
    Create side-by-side bar charts for loan acceptance vs rejection percentages 
    for multiple categorical features in one figure.

    Args:
        data: DataFrame containing loan data.
        categorical_features: List of categorical features to plot.
        figsize: Tuple specifying the figure size.

    Returns:
        None. Displays the plots.
    """
    # Set up the figure and axes
    n_features = len(categorical_features)
    n_cols = 2  # Number of columns
    n_rows = (n_features + 1) // n_cols  # Number of rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22,15))
    axes = axes.flatten()

    for i, feature in enumerate(categorical_features):
        # Calculate accepted and rejected percentages
        summary = data.groupby(feature).agg(
            accepted_loans=('loan_status', lambda x: (x == 1).sum()),
            total_loans_applied=('loan_status', 'size')
        )
        summary['accepted_percentage'] = (
            summary['accepted_loans'] / summary['total_loans_applied'] * 100
        )
        summary['rejected_percentage'] = 100 - summary['accepted_percentage']

        # Positioning for side-by-side bars
        categories = summary.index
        x = np.arange(len(categories))  # Label positions
        width = 0.35  # Width of bars

        # Current axis
        ax = axes[i]

        # Plotting side-by-side bars
        bar1 = ax.bar(
            x - width / 2,
            summary['accepted_percentage'],
            width,
            label='Accepted %',
            color='limegreen',
            edgecolor='black'
        )
        bar2 = ax.bar(
            x + width / 2,
            summary['rejected_percentage'],
            width,
            label='Rejected %',
            color='salmon',
            edgecolor='black'
        )

        # Adding percentage labels inside the bars
        for bar in bar1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='black')

        for bar in bar2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='black')

        # Formatting
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Percentage', fontsize=10)
        ax.set_title(f'Loan Acceptance vs Rejection by {feature}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.legend(fontsize=9)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img





def plot_distributions(data: pd.DataFrame) -> None:
    """
    Generates various distribution plots and saves the figure.

    Args:
        data (pd.DataFrame): Input dataset.
        output_path (str): Path to save the generated plot.
    """
    # Set up subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Age group distribution
    data['age_group'] = (data['person_age'] // 10) * 10
    sns.countplot(x='age_group', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Age Groups')
    axes[0, 0].set_xlabel('Age Group (10-year intervals)')
    axes[0, 0].set_ylabel('Count')

    # Income distribution
    income_bins = [0, 50000, 100000, 150000, 200000, float('inf')]
    income_labels = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K+']
    data['income_category'] = pd.cut(data['person_income'], bins=income_bins, labels=income_labels, right=False)
    sns.countplot(x='income_category', data=data, order=income_labels, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Person Income')
    axes[0, 1].set_xlabel('Income')
    axes[0, 1].set_ylabel('Count')

    # Employment length distribution
    emp_length_bins = [-1, 0, 5, 10, 15, 20, float('inf')]
    emp_length_labels = ['<1 year', '1-5 years', '6-10 years', '11-15 years', '16-20 years', '20+ years']
    data['emp_length_category'] = pd.cut(data['person_emp_length'], bins=emp_length_bins, labels=emp_length_labels, right=False)
    sns.countplot(x='emp_length_category', data=data, order=emp_length_labels, ax=axes[0, 2])
    axes[0, 2].set_title('Distribution of Employment Length')
    axes[0, 2].set_xlabel('Employment Length')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xticklabels(emp_length_labels, rotation=45)

    # Loan amount distribution
    loan_amnt_bins = [0, 5000, 10000, 15000, 20000, 25000, float('inf')]
    loan_amnt_labels = ['0-5K', '5K-10K', '10K-15K', '15K-20K', '20K-25K', '25K+']
    data['loan_amnt_category'] = pd.cut(data['loan_amnt'], bins=loan_amnt_bins, labels=loan_amnt_labels, right=False)
    sns.countplot(x='loan_amnt_category', data=data, order=loan_amnt_labels, ax=axes[0, 3])
    axes[0, 3].set_title('Distribution of Loan Amount')
    axes[0, 3].set_xlabel('Loan Amount Category')
    axes[0, 3].set_ylabel('Count')

    # Credit history length
    sns.histplot(data['cb_person_cred_hist_length'], bins=20, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Credit History Length')
    axes[1, 0].set_xlabel('Credit History Length (Years)')
    axes[1, 0].set_ylabel('Count')

    # Loan interest rate
    sns.histplot(data['loan_int_rate'], bins=20, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Loan Interest Rate')
    axes[1, 1].set_xlabel('Loan Interest Rate')
    axes[1, 1].set_ylabel('Count')

    # Loan percent of income
    sns.histplot(data['loan_percent_income'], bins=20, ax=axes[1, 2])
    axes[1, 2].set_title('Distribution of Loan Amount Percentage of Income')
    axes[1, 2].set_xlabel('Loan Amount Percentage of Income')
    axes[1, 2].set_ylabel('Count')

    # Hide unused subplot
    axes[1, 3].axis('off')

    # Save and display
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the raw data.

    Args:
        data (pd.DataFrame): The raw input data.

    Returns:
        pd.DataFrame: Data without duplicates.
    """
    return data.drop_duplicates()

def plot_categorical_distributions(data: pd.DataFrame) -> None:
    """
    Plots the distribution of categorical columns in the dataset.

    Args:
        train (pd.DataFrame): The input dataset containing categorical columns.
    """
    # Select categorical columns
    categorical_cols = data.select_dtypes(include='object').columns

    # Number of categorical columns and layout settings
    num_plots = len(categorical_cols)
    rows = (num_plots + 3) // 4  # Calculate rows dynamically
    cols = min(num_plots, 4)  # Max columns per row is 4

    # Create a figure with adjusted size
    fig = plt.figure(figsize=(20,12 ))

    # Generate count plots for each categorical column
    for i, col in enumerate(categorical_cols):
        plt.subplot(rows, cols, i + 1)
        sns.countplot(x=data[col])
        plt.title(f'Distribution of {col}', fontsize=12)
        plt.xticks(rotation=45, ha='right')

    # Adjust layout and show the plots
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img


def plot_categorical_relations(
    df: pd.DataFrame, 
    categorical_features: list, 
    #continuous_feature: str, 
    hue_feature: str
) -> None:
    """
    Plots the relationship between a continuous feature and multiple categorical features
    using a swarm plot.

    Args:
        df (pd.DataFrame): The input dataset containing the features.
        categorical_features (list): A list of categorical features to be plotted.
        continuous_feature (str): The continuous feature to be plotted on the y-axis.
        hue_feature (str): The feature used for color grouping (hue).
    """
    for cat_feature in categorical_features:
        fig= plt.figure(figsize=(12, 6))
        sns.swarmplot(
            data=df.sample(1000),  # Sample 1000 rows for the plot
            x=cat_feature,          # Categorical feature (on the x-axis)
            y=df['loan_amnt'],   # Continuous feature (on the y-axis)
            hue=hue_feature,        # Categorical feature (hue for color grouping)
            palette='viridis'       # Color palette
        )
        plt.title(f'Relation of Loan_amount and {cat_feature} by {hue_feature}')
        fig.tight_layout()
        img=plt.gcf()
    # Return the figure object
        return img
    
    
    
    
    
def plot_categorical_relations_grade(
    df: pd.DataFrame, 
    categorical_features: list, 
    #continuous_feature: str, 
    hue_feature: str
) -> None:
    """
    Plots the relationship between a continuous feature and multiple categorical features
    using a swarm plot.

    Args:
        df (pd.DataFrame): The input dataset containing the features.
        categorical_features (list): A list of categorical features to be plotted.
        continuous_feature (str): The continuous feature to be plotted on the y-axis.
        hue_feature (str): The feature used for color grouping (hue).
    """
    for cat_feature in categorical_features:
        fig= plt.figure(figsize=(12, 6))
        sns.swarmplot(
            data=df.sample(1000),  # Sample 1000 rows for the plot
            x=df['loan_grade'],          # Categorical feature (on the x-axis)
            y=df['loan_amnt'],   # Continuous feature (on the y-axis)
            hue=hue_feature,        # Categorical feature (hue for color grouping)
            palette='viridis'       # Color palette
        )
        plt.title(f'Relation of Loan_amount and {cat_feature} by {hue_feature}')
        fig.tight_layout()
        img=plt.gcf()
    # Return the figure object
        return img
    
    
    



def plot_histograms_kde(
    df: pd.DataFrame, 
    hist_columns: list, 
    hue_column: str
) -> None:
    """
    Plots histograms for specified numerical columns with hue differentiation.
    
    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        hist_columns (list): List of columns to plot histograms for.
        hue_column (str): The categorical column to use as the hue (e.g., 'loan_status').
    """
    # Set up the grid layout
    num_plots = len(hist_columns)
    rows = (num_plots + 2) // 3  # Arrange in 3 columns
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()  # Flatten to iterate over all axes

    # Loop through the columns and create histograms
    for i, col in enumerate(hist_columns):
        sns.histplot(
            data=df,
            x=col,
            hue=hue_column,
            palette='viridis',
            kde=True,
            ax=axes[i]  # Specify the subplot axis
        )
        axes[i].set_title(f'Histogram of {col}', fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    fig.tight_layout()
    img=plt.gcf()
    # Return the figure object
    return img
    