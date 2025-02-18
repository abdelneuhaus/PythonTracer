# Test file for spline fitting module

import cv2
import numpy as np
from scipy.signal import convolve2d

from python_tracer.SplineFitterSMAP import set_parameters
from python_tracer.SplineFitterSMAP import gaussian_filter
from python_tracer.SplineFitterSMAP import filter2
from python_tracer.SplineFitterSMAP import difference_of_gaussians
from python_tracer.SplineFitterSMAP import maximumfindcall

##################################################
def test_set_parameters():
    """
    Test set_parameters() function.
    """
    parameters = set_parameters('stack', 'calib_file', 498, 0.12, 1.2, 15, 13, './outputSMAP.csv', 160)
    assert isinstance(parameters, dict), "Le retour de set_parameters() n'est pas un dictionnaire"
    
    keys = {
        "imagefile", "calibfile", "offset", "conversion", "peakfilter", "peakcutoff", "roifit",
        "bidirectional", "mirror", "status", "outputfile", "outputformat", "pixelsize",
        "loader", "mij", "backgroundmode", "isscmos", "scmosfile", "preview"
    }
    assert set(parameters.keys()) == keys, f"A key is missing. Should have : {parameters.keys()}"
    
    expected_values = {
        "imagefile": 'stack',
        "calibfile": 'calib_file',
        "offset": 498,
        "conversion": 0.12,
        "peakfilter": 1.2,
        "peakcutoff": 15,
        "roifit": 13,
        "outputfile": './outputSMAP.csv',
        "outputformat": 'csv',
        "pixelsize": 160,
        "bidirectional": 0,
        "mirror": 0,
        "status": [],
        "loader": 1,
        "mij": [],
        "backgroundmode": 'Difference of Gaussians (fast)',
        "isscmos": 0,
        "scmosfile": '',
        "preview": 0
    }

    for key, expected in expected_values.items():
        assert parameters[key] == expected, f"Incorrect value of {key}: {expected} is expected, got {parameters[key]}."




##################################################
def test_gaussian_filter():
    """
    Test gaussian_filter() function.
    """
    HSIZE = 5
    SIGMA = 1.0
    kernel = gaussian_filter(HSIZE, SIGMA)

    assert isinstance(kernel, np.ndarray), "Filter is not numpy array"
    assert kernel.shape == (HSIZE, HSIZE), f"Incorrect size : {(HSIZE, HSIZE)} expected, got {kernel.shape}"
    assert np.allclose(kernel, kernel.T), "Gaussian kernel non symmetrical"
    assert np.isclose(kernel.sum(), 1, atol=1e-5), f"Kernel coefficients sum is {kernel.sum()}, expected value close to 1"




##################################################
def test_filter2():
    """
    Test filter2() function.
    """
    # Identity case (simple case)
    x = np.array([[1, 2], [3, 4]])
    b = np.array([[0, 0], [0, 1]])
    expected = convolve2d(x, np.rot90(b, 2), mode='same')
    assert np.allclose(filter2(b, x), expected)
    
    # Mean filter case
    b = np.ones((3, 3)) / 9
    expected = convolve2d(x, np.rot90(b, 2), mode='same')
    assert np.allclose(filter2(b, x), expected)




##################################################
def test_difference_of_gaussians():
    """
    Test difference_of_gaussians() function.
    """
    # Test image
    imphot = np.ones((10, 10))
    peakfilter = 1.0
    result = difference_of_gaussians(imphot, peakfilter)

    # Shape checking
    assert result.shape == imphot.shape

    # Verify values sum close to 0
    assert np.allclose(np.sum(result), 0, atol=1e-6)




##################################################
def test_maximumfindcall():
    """
    Test maximumfindcall() function.
    """
    # One center in the image (center)
    imin = np.zeros((5, 5))
    imin[2, 2] = 10
    result = maximumfindcall(imin)
    expected = np.array([[2, 2, 10]])
    assert np.array_equal(result, expected)

    # No maxima
    imin = np.ones((5, 5))
    result = maximumfindcall(imin)
    assert result.size == 0