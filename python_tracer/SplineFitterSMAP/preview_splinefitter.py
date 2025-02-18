import math
import numpy as np
import scipy.io as io
import tifffile as tif
from napari.utils.notifications import show_info

from SplineFitterSMAP.difference_of_gaussians import difference_of_gaussians
from SplineFitterSMAP.maximumfindcall import maximumfindcall

def previews_splinefitter(parameters, number_frames=1):
    """
    Cspline fitter. Reference: https://github.com/jries/fit3Dcspline/blob/master/simplefitter_cspline.m

    Args:
        parameters (dict): dictionary from parameters (SMAP output file)

    Returns:
        None
    """

    if parameters['calibfile']:
        cal = io.loadmat(parameters['calibfile']) 
        parameters['dz'] = cal['SXY']['cspline'][0,0]['dz'][0][0][0][0]
        parameters['z0'] = cal['SXY']['cspline'][0,0]['z0'][0][0][0][0]
        parameters['coeff'] = cal['SXY']['cspline'][0,0]['coeff'][0][0][0][0]
        if isinstance(parameters['coeff'], (list, tuple)):
            parameters['coeff'] = parameters['coeff'][0]
    else:
        print('3D calibration file could not be loaded. Using Gaussian fitter instead.')    # not implemented. Should always have calibfile
        parameters['isspline'] = False

    parameters['isspline'] = True

    # dx = half or the ROI (roifit=13 means dx=6)
    parameters['dx'] = math.floor(parameters['roifit']/2)

    img = tif.imread(parameters['imagefile'])[0:number_frames] # Load only the first number_frames frames
    frame_index = 0
    stock_imphot = []
    stock_maxgood = []

    show_info("Fast Fitting for Preview")
    for f in img:
        frame_index += 1
        size_img = f.shape
        imphot = (f.astype(np.float32) - parameters['offset']) * parameters['conversion']   # Pixels to photons conversion
        stock_imphot.append(imphot)
        # Detection filtrage
        impf = difference_of_gaussians(imphot, parameters['peakfilter'])
        maxima = maximumfindcall(impf)  # find local maxima
        indmgood = maxima[:, 2] > parameters['peakcutoff']  # filter out maxima below cutoff
        # filter ROI that are too close to inital image edges
        indmgood &= (maxima[:, 0] > parameters['dx']) & (maxima[:, 0] <= size_img[1] - parameters['dx'])
        indmgood &= (maxima[:, 1] > parameters['dx']) & (maxima[:, 1] <= size_img[0] - parameters['dx'])
        maxgood = maxima[indmgood, :]
        stock_maxgood.append(maxgood)

    return img, stock_imphot, stock_maxgood
