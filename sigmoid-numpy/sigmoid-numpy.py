import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    np_array = np.array(x)
    return 1/(1+ np.exp(-1*np_array))
    # Write code here
