import numpy as np

from SplineFitterSMAP.mleFit import mleFit_LM

def fit_spline(imstack, peakcoordinates, parameters, varstack):
    """
    Perform the spline fitting of ROIs in imstack.
    Args:
        imstack (np.ndarray): stack of images (ROIs)
        peakcoordinates (np.ndarray): coordinates of the detected PSFs
        parameters (dict): parameters dictionary
        varstack (np.ndarray): variance map if sCMOS camera. Else, 0 or None.
    Returns:
        np.ndarray: results of the spline fitting
    """
    z0 = 0
    zstart = 0
    zstart = (z0+parameters['coeff'].shape[2]/2)
    fitpar = np.float32(parameters['coeff'])
    param_cspline, crlbs, loglikehood = mleFit_LM(imstack, 30, fitpar, varstack, zstart)

    results = np.zeros((imstack.shape[2], 12))
    results[:, 0] = peakcoordinates[:, 2]
    results[:, 1] = param_cspline[:, 0] - parameters['dx'] + peakcoordinates[:, 0]  # x
    results[:, 2] = param_cspline[:, 1] - parameters['dx'] + peakcoordinates[:, 1] # y
    results[:, 3] = (param_cspline[:, 4] - parameters['z0']) * parameters['dz'] # z
    results[:, 4:6] = param_cspline[:, 2:4] # photons, background
    results[:, 6:8] = np.real(np.sqrt(crlbs[:, [1, 0]])) # x, y CRLBs
    results[:, 8] = np.real(np.sqrt(crlbs[:, 4] * parameters['dz'])) # z CRLB
    results[:, 9:11] = np.real(np.sqrt(crlbs[:, 2:4])) # photons, background CRLBs
    results[:, 11] = loglikehood # Log likelihood

    return results