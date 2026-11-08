import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    
    # Write code here
    y_pred = np.array(y_pred)
    row_indices = np.arange(len(y_true))
    correct_class_per_sample = y_pred[row_indices,y_true]
    logs = np.log(correct_class_per_sample)
    return float(-1*np.mean(logs))
    
    
    