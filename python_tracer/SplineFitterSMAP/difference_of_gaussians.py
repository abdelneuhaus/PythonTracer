import numpy as np
from .filter2 import filter2
from .fspecial import gaussian_filter  # la même que votre code

def difference_of_gaussians(imphot, peakfilter):
    """
    This function computes differences of gaussian (DoG) of an image. Traduction of https://github.com/jries/SMAP/blob/master/fit3Dcspline/simplefitter_cspline.m
    Args: 
      imphot: array, one image of the stack
      peakfilter: float, peak value to use
    Return:
      impf: array, processed image
    """

    # Determine size of the kernel
    rsize = 2 * int(np.ceil(3 * peakfilter)) + 1

    # Build 2 gaussian kernels
    sigma1 = peakfilter
    sigma2 = max(1, 2.5 * peakfilter)

    gauss1 = gaussian_filter(rsize, sigma1)  # 2D gauss
    gauss2 = gaussian_filter(rsize, sigma2)

    # Difference of Gaussians
    hdog = gauss1 - gauss2
    dog_input = imphot - np.min(imphot[:, 0])

    # Apply 2D convolution
    impf = filter2(hdog, dog_input, shape='same')

    return impf
