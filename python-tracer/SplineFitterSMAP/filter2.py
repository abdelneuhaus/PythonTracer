import numpy as np
from scipy.signal import convolve2d

def filter2(b, x, shape='same'):
    """
    Two-dimensional digital filter.
    
    Parameters:
    b : ndarray
        The 2-D FIR filter.
    x : ndarray
        The input data to filter.
    shape : {'same', 'valid', 'full'}, optional
        Defines the size of the output, default is 'same'.
    
    Returns:
    y : ndarray
        The filtered data.
    """
    
    # Ensure that the inputs are floats
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Rotate the filter matrix by 180 degrees
    stencil = np.rot90(b, 2)
    
    # Perform the convolution
    y = convolve2d(x, stencil, mode=shape)
    
    return y