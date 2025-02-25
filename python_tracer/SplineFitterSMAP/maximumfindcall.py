import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects

def maximumfindcall(imin):
    """
    This function finds local maxima in a 2D image using similar approch to Neubeck&Gool algorithm.
    Adapted from https://github.com/jries/fit3Dcspline/blob/master/shared/maximumfindcall.m
    Args:
        imin: np.array, input image
    Return:
        maximaout: array, contains [x,y, intensity] for each maxima
    """
    # Apply maximum filter to find local maxima
    neighborhood_size = 3
    data_max = maximum_filter(imin, neighborhood_size)
    maxima = (imin == data_max)
    
    # Remove image border
    maxima[0, :] = maxima[-1, :] = maxima[:, 0] = maxima[:, -1] = False
    
    # Get local maxima coordinates (similar to do_fit_and_gallery)
    labeled, num_objects = label(maxima)
    slices = find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) // 2
        y_center = (dy.start + dy.stop - 1) // 2
        x.append(x_center)
        y.append(y_center)    
    intensities = imin[y, x]
    
    # Output data
    maximaout = np.array(list(zip(x, y, intensities)))
    
    return maximaout