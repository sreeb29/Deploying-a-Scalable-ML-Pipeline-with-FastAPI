import pytest
# TODO: add necessary import
from ml.model import compute_model_metrics
# TODO: implement the first test. Change the function name and input as needed
def test_compute_model_metrics_perfect_prediction():
    """
    # Test metrics when predictions perfectly match true labels.
    """
    # Your code here
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Perfect predictions should result in 1.0 for all metrics
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics_partial_match():
    """
    # Test metrics with a mix of true positives, false positives, and false negatives.
    """
    # Your code here
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # TP=1, FP=1, FN=1. Precision = 1/2, Recall = 1/2, F1 = 1/2
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_zero_division():
    """
    # Test that zero_division=1 handles edges cases like no positive predictions.
    """
    # Your code here
    y = np.array([1, 1, 1])
    preds = np.array([0, 0, 0])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # Precision is 0/0 -> falls back to 1.0 due to zero_division=1
    assert precision == 1.0 
    assert recall == 0.0
