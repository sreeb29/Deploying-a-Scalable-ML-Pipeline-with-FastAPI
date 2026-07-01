import pytest
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from ml.model import compute_model_metrics, train_model

# add necessary import

# implement the first test. Change the function name and input as needed
def test_dataset_shape():
    # Example test for dataset sizes and types (Idea 4)
    
    try:
        df = pd.read_csv('data/cleaned_census_data.csv')
        print("Dataset file found.")
    except FileNotFoundError:
        pytest.fail("Dataset file not found. Please verify the path.")
    
    # Check if the DataFrame has rows
    assert df.shape[0] > 0, "Dataset should have more than 0 rows."
    # Check if it is a pandas DataFrame
    assert isinstance(df, pd.DataFrame), "Data should be loaded as a pandas DataFrame."

# implement the second test. Change the function name and input as needed
def test_ml_function_output_type():
    """
    # testing to verify ML function returns expected type on 'train_model' function in ml code that outputs a model.
    """
    # Your code here
    # Creating some dummy training data
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    
    model = train_model(X_train, y_train)
    
    # Verify the output is a trained scikit-learn model
    assert isinstance(model, RandomForestClassifier), "Function should return a RandomForestClassifier instance."



# implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # testing to verify computing metrics returns the expected value.
    """
    # Your code here
    y_sample = [0, 1, 1, 0]
    preds_sample = [0, 1, 0, 0]
    
    precision, recall, fbeta = compute_model_metrics(y_sample, preds_sample)
    
    # Verify the results return valid floats
    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(fbeta, float), "F-beta should be a float."

def test_compute_model_metrics_partial_match():
    """Test metrics with a mix of true positives, false positives, and false negatives."""
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # TP=1, FP=1, FN=1. Precision = 1/2, Recall = 1/2, F1 = 1/2
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)

def test_compute_model_metrics_zero_division():
    """Test that zero_division=1 handles edges cases like no positive predictions."""
    y = np.array([1, 1, 1])
    preds = np.array([0, 0, 0])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # Precision is 0/0 -> falls back to 1.0 due to zero_division=1
    assert precision == 1.0 
    assert recall == 0.0
