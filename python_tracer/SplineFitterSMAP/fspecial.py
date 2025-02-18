import cv2

def gaussian_filter(HSIZE, SIGMA):
    k1d = cv2.getGaussianKernel(HSIZE, SIGMA)
    k2d = k1d @ k1d.T
    return k2d