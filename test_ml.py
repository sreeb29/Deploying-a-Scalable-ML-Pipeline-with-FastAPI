import pytest
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from ml.model import compute_model_metrics, train_model

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_dataset_shape():
    # Example test for dataset sizes and types (Idea 4)
    # Replace 'data/clean_data.csv' with your actual data loading/fixture
    try:
        df = pd.read_csv('data/cleaned_census_data.csv')
        print("Dataset file found.")
    except FileNotFoundError:
        pytest.fail("Dataset file not found. Please verify the path.")
    
    # Check if the DataFrame has rows
    assert df.shape[0] > 0, "Dataset should have more than 0 rows."
    # Check if it is a pandas DataFrame
    assert isinstance(df, pd.DataFrame), "Data should be loaded as a pandas DataFrame."


# TODO: implement the second test. Change the function name and input as needed
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



# TODO: implement the third test. Change the function name and input as needed
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

