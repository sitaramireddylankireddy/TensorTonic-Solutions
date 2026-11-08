import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    # x = [[1, 2, 3], [0, 0, 0]]
    # print(np.array(x))
    np_array = np.array(x) - np.max(x, axis=-1, keepdims=True)
    # print(np_array)
    exp_array = np.exp(np_array)
    denominator = exp_array.sum(axis=-1, keepdims=True)
    return exp_array/denominator