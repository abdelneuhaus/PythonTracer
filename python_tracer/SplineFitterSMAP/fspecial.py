import cv2

def gaussian_filter(HSIZE, SIGMA):
    """
    This function creates a gaussian kernel.
    Args:
        HSIZE: int, size of kernel
        SIGMA: float, sigma to generate the kernel
    Returns:
        k2d: array, gaussian kernel
    """
    k1d = cv2.getGaussianKernel(HSIZE, SIGMA)
    k2d = k1d @ k1d.T   # transpose
    return k2d