import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the training data based on specified conditions:
    - `person_age` < 90
    - `person_income` < 1e6
    - `person_emp_length` < 60

    Args:
        df_train: Input training DataFrame.

    Returns:
        Filtered DataFrame.
    """
    data= data.drop(['id'], axis=1)
    data_1 = data[data['person_age'] < 90].reset_index(drop=True)
    data_2 = data_1[data_1['person_income'] < 1e6].reset_index(drop=True)
    df2 = data_2[data_2['person_emp_length'] < 60].reset_index(drop=True)

    return df2


def feature_target_split(data: pd.DataFrame):
    
    
    features = data.drop(columns='loan_status', axis =1)
    target = data['loan_status']
    return features, target
    
    

    
    
def train_test_df_split(features: pd.DataFrame, target: pd.Series, test_size: float , random_state: float):
    """
    Splits the data into training and testing sets.

    Args:
        features (pd.DataFrame): The feature matrix (X).
        target (pd.Series): The target variable (y).
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def one_hot_encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Concatenates training and test datasets, applies one-hot encoding, 
    and splits the datasets back into training and test sets.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        cat_cols (list): List of categorical columns to be one-hot encoded.

    Returns:
        tuple: One-hot encoded X_train and X_test datasets.
    """
    # Concatenate X_train and X_test
    combined_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    cat_cols= combined_data.select_dtypes(['object']).columns
    # One-hot encode the categorical columns
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_data = onehot_encoder.fit_transform(combined_data[cat_cols])
    
    # Convert encoded data to DataFrame
    encoded_columns = onehot_encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Drop the original categorical columns and add the encoded ones
    combined_data_encoded = combined_data.drop(columns=cat_cols).join(encoded_df)
    
    # Split the data back into X_train and X_test
    X_train_encoded = combined_data_encoded.iloc[:len(X_train), :]
    X_test_encoded = combined_data_encoded.iloc[len(X_train):, :]

    return X_train_encoded, X_test_encoded




def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, stand_col: list):
    """
    Scales the numerical columns in the training and testing datasets.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        stand_col (list): List of numerical columns to be standardized.

    Returns:
        tuple: Scaled X_train and X_test datasets.
    """
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train[stand_col])
    X_test_scaled = scaler.transform(X_test[stand_col])

    # Convert back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=stand_col, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=stand_col, index=X_test.index)

    # Replace original columns with scaled columns
    X_train_final = X_train.drop(columns=stand_col).join(X_train_scaled_df)
    X_test_final = X_test.drop(columns=stand_col).join(X_test_scaled_df)

    return X_train_final, X_test_final


import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Dict
import pickle


def fit_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    models: List[str],
    model_params: Dict[str, Dict],
    #cat_cols: List[str],
    stand_col: List[str],
) -> Dict[str, str]:
    """
    Fits models and saves their pipelines for later evaluation.

    Args:
        x_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        models (List[str]): List of model names.
        model_params (Dict[str, Dict]): Parameters for each model.
        cat_cols (List[str]): List of categorical columns for one-hot encoding.
        stand_col (List[str]): List of numerical columns for standard scaling.

    Returns:
        Dict[str, str]: A dictionary with model names and paths to their pickled pipelines.
    """
    model_paths = {}

    for model_name in models:
        # Instantiate model with parameters if available
        model_class = globals()[model_name]
        params = model_params.get(model_name, {})
        model = model_class(**params)

        #combined_data = pd.concat([x_train, X_test], axis=0, ignore_index=True)
        cat_cols= x_train.select_dtypes(['object']).columns
        # Create a column transformer for encoding and scaling
        coltrans = ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("scale", StandardScaler(), stand_col),
            ],
            remainder="passthrough",
        )

        # Create a pipeline with the column transformer and model
        pipeline = make_pipeline(coltrans, model)

        # Fit the pipeline on the training data
        

        # Save the pipeline
        path = f"data/04_models/pipeline.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)

        model_paths[model_name] = path

        print(f"Model {model_name} has been trained and saved to {path}.")

    return model_paths

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd
from typing import Dict

def evaluate_models(
    model_paths: Dict[str, str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates models and generates predictions.

    Args:
        model_paths (Dict[str, str]): Dictionary of model names and their pipeline paths.
        x_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target variable.
        x_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): True labels for the test set.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing CV scores, test predictions, and evaluation metrics for each model.
    """
    results = {}

    for model_name, path in model_paths.items():
        # Load the saved pipeline
        with open(path, "rb") as f:
            pipeline = pickle.load(f)

        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=10)

        # Predict on the test set
        y_pred = pipeline.predict(x_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Store results
        results[model_name] = {
            "CV Mean": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Predictions": y_pred.tolist(),
        }

        print(f"Model: {model_name}")
        print(f"CV Mean: {cv_scores.mean():.4f}, CV Std: {cv_scores.std():.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print("-" * 50)

    return results

