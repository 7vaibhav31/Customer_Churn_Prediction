"""
Preprocessing module for Customer Churn Prediction.
Handles data loading, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data(csv_path):
    """Load CSV data from the given path."""
    df = pd.read_csv(csv_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def clean_data(df):
    """
    Clean data by removing unnecessary columns.
    Remove: RowNumber, CustomerId, Surname
    """
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=columns_to_drop)
    print(f"Data cleaned. Shape after dropping columns: {df.shape}")
    return df


def split_features_and_target(df, target_col='Exited'):
    """Split data into features (X) and target (y)."""
    x = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"Features shape: {x.shape}, Target shape: {y.shape}")
    return x, y


def identify_column_types(x_train):
    """Identify categorical and numerical columns."""
    categorical_cols = ['Gender', 'Geography']
    numerical_cols = [col for col in x_train.columns if col not in categorical_cols]
    
    print("Categorical Columns:", categorical_cols)
    print("Numerical Columns:", numerical_cols)
    
    return categorical_cols, numerical_cols


def create_preprocessor(categorical_cols, numerical_cols):
    """
    Create a ColumnTransformer with OneHotEncoder for categorical 
    and StandardScaler for numerical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    print("Preprocessor created successfully.")
    return preprocessor


def preprocess_data(x_train, x_test, preprocessor):
    """Fit preprocessor on training data and transform both sets."""
    preprocessor.fit(x_train)
    
    x_train_transformed = preprocessor.transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)
    
    print(f"Training data transformed. Shape: {x_train_transformed.shape}")
    print(f"Test data transformed. Shape: {x_test_transformed.shape}")
    
    return x_train_transformed, x_test_transformed


def save_preprocessor(preprocessor, filepath='preprocessor.joblib'):
    """Save the fitted preprocessor to a file."""
    joblib.dump(preprocessor, filepath)
    print(f"Preprocessor saved to {filepath}")


def load_preprocessor(filepath='preprocessor.joblib'):
    """Load the preprocessor from a file."""
    if os.path.exists(filepath):
        preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor
    else:
        raise FileNotFoundError(f"Preprocessor file not found at {filepath}")


def preprocess_pipeline(csv_path, test_size=0.2, random_state=42, save_preprocessor_flag=True):
    """
    Complete preprocessing pipeline:
    Load -> Clean -> Split -> Identify columns -> Create preprocessor -> Transform -> Save
    """
    # Load and clean data
    df = load_data(csv_path)
    df = clean_data(df)
    
    # Split features and target
    x, y = split_features_and_target(df)
    
    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    print(f"Train-test split: X_train={x_train.shape}, X_test={x_test.shape}")
    
    # Identify column types
    categorical_cols, numerical_cols = identify_column_types(x_train)
    
    # Create preprocessor
    preprocessor = create_preprocessor(categorical_cols, numerical_cols)
    
    # Preprocess data
    x_train_transformed, x_test_transformed = preprocess_data(x_train, x_test, preprocessor)
    
    # Save preprocessor if needed
    if save_preprocessor_flag:
        save_preprocessor(preprocessor)
    
    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'x_train_transformed': x_train_transformed,
        'x_test_transformed': x_test_transformed,
        'preprocessor': preprocessor,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }


if __name__ == "__main__":
    # Example usage
    csv_path = "Churn_Modelling.csv"  # Update with your csv path
    result = preprocess_pipeline(csv_path)
    print("\nPreprocessing complete!")
    print(f"Training set shape: {result['x_train_transformed'].shape}")
    print(f"Test set shape: {result['x_test_transformed'].shape}")
