import pytest
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from ml.model import compute_model_metrics, train_model

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
@pytest.fixture(scope="session")
def census_dataframe():
    # Replace with your actual path to the clean census.csv file
    df = pd.read_csv("./data/census.csv")
    return df

def test_dataset_shape(census_dataframe):
    """
    # Testing for dataset size and type.
    """
    # Your code here
    assert census_dataframe.isnull().sum().sum() == 0, "Dataset contains unexpected null values."



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

