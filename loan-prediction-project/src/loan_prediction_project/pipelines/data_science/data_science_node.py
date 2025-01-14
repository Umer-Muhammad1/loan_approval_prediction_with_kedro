import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import logging



def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    
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
    
 
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def one_hot_encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    
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






def train_evaluate_xgb(x_train, y_train, x_test, y_test, params):
   
    logger = logging.getLogger(__name__)

    # Initialize the XGBClassifier with parameters
    xgb = XGBClassifier(**params)

    # Perform cross-validation
    logger.info("Starting cross-validation...")
    cv_scores = cross_val_score(xgb, x_train, y_train, cv=10, scoring='roc_auc')
    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()

    logger.info(f"Cross-validation AUC-ROC scores: {cv_scores}")
    logger.info(f"Mean CV AUC-ROC score: {mean_cv_score:.4f}")
    logger.info(f"Std CV AUC-ROC score: {std_cv_score:.4f}")

    # Train the model
    logger.info("Training the XGB model...")
    xgb.fit(x_train, y_train)

    # Evaluate on training data
    y_train_pred = xgb.predict(x_train)
    y_train_proba = xgb.predict_proba(x_train)[:, 1]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auc_roc = roc_auc_score(y_train, y_train_proba)
    logger.info(f"Training Accuracy: {train_accuracy:.4f}")
    logger.info(f"Training AUC-ROC: {train_auc_roc:.4f}")

    # Evaluate on testing data
    y_test_pred = xgb.predict(x_test)
    y_test_proba = xgb.predict_proba(x_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc_roc = roc_auc_score(y_test, y_test_proba)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test AUC-ROC: {test_auc_roc:.4f}")

    # Return results as a dictionary
    return {
        "model": xgb,
        "cv_scores_auc_roc": cv_scores.tolist(),
        "mean_cv_auc_roc": mean_cv_score,
        "std_cv_auc_roc": std_cv_score,
        "train_accuracy": train_accuracy,
        "train_auc_roc": train_auc_roc,
        "test_accuracy": test_accuracy,
        "test_auc_roc": test_auc_roc,
    }

