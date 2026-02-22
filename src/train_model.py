"""
Model training module for Customer Churn Prediction.
Defines and trains an MLP Perceptron model using TensorFlow/Keras.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import warnings

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

# Support both package-style import (when running as a module) and
# standalone import (when running the script directly).
try:
    from src.preprocessing import preprocess_pipeline
except Exception:
    from preprocessing import preprocess_pipeline

warnings.filterwarnings('ignore')


def create_model(input_dim=13):
    """
    Create and compile a Sequential Keras model (MLP Perceptron).
    """
    model = Sequential()
    model.add(Dense(units=6, activation='relu', input_dim=input_dim))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def make_model():
    return create_model(input_dim=13)


def train_model(x_train, x_test, y_train, y_test, preprocessor, epochs=100, verbose=1):
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    input_dim = preprocessor.fit_transform(x_train).shape[1]
    keras_classifier = KerasClassifier(model=make_model, epochs=epochs, batch_size=32, verbose=verbose)
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', keras_classifier)])
    print(f"\nTraining MLP Perceptron for {epochs} epochs...")
    pipeline.fit(x_train, y_train)
    print("\nTraining completed!")
    x_test_preprocessed = preprocessor.transform(x_test)
    keras_classifier_model = pipeline.named_steps['classifier']
    y_pred = keras_classifier_model.predict(x_test_preprocessed)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ON TEST SET")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*60)
    return {'pipeline': pipeline, 'preprocessor': preprocessor, 'keras_model': keras_classifier_model.model_, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'input_dim': input_dim, 'y_pred': y_pred}


def save_model(keras_model, filepath='keras_model.keras'):
    keras_model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='keras_model.keras'):
    if os.path.exists(filepath):
        model = tensorflow.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {filepath}")


def make_predictions(new_data_df, preprocessor_path='preprocessor.joblib', model_path='keras_model.keras'):
    preprocessor = joblib.load(preprocessor_path)
    model = tensorflow.keras.models.load_model(model_path)
    new_data_preprocessed = preprocessor.transform(new_data_df)
    raw_predictions = model.predict(new_data_preprocessed)
    predicted_classes = (raw_predictions > 0.5).astype(int).flatten()
    return predicted_classes, raw_predictions


def main():
    import kagglehub
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("rjmanoj/credit-card-customer-churn-prediction")
    csv_path = f"{path}/Churn_Modelling.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    print("\n" + "="*60)
    print("STARTING DATA PREPROCESSING")
    print("="*60)
    preprocess_result = preprocess_pipeline(csv_path)
    x_train = preprocess_result['x_train']
    x_test = preprocess_result['x_test']
    y_train = preprocess_result['y_train']
    y_test = preprocess_result['y_test']
    preprocessor = preprocess_result['preprocessor']
    result = train_model(x_train, x_test, y_train, y_test, preprocessor, epochs=100)
    print("\n" + "="*60)
    print("SAVING MODEL AND PREPROCESSOR")
    print("="*60)
    save_model(result['keras_model'], 'keras_model.keras')
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
