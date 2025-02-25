import math
import numpy as np
import scipy.io as io
import tifffile as tif
import matplotlib.pyplot as plt
from napari.utils.notifications import show_info

from .difference_of_gaussians import difference_of_gaussians
from .fitspline import fit_spline
from .maximumfindcall import maximumfindcall
from .save_results_python import save_results_python
from .save_as_locpalmtracer import save_as_locpalmtracer

def cspline_fitter(parameters, preview=False):
    """
    Cspline fitter. Reference: https://github.com/jries/fit3Dcspline/blob/master/simplefitter_cspline.m

    Args:
        parameters (dict): dictionary from parameters (SMAP output file)
        preview (bool): don't use
    Returns:
        None
    """

    subimages = []  #  PSFs crop (ROIs)
    peakcoords = [] # (x, y, frame)

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

    img = tif.imread(parameters['imagefile'])
    frame_index = 0

    show_info("Detecting and cropping PSFs")
    for f in img:
        if preview:
            plt.ion() # Activate interactive mode to preview in live
        frame_index += 1
        size_img = f.shape
        imphot = (f.astype(np.float32) - parameters['offset']) * parameters['conversion']   # Pixels to photons conversion

        # Detection filtrage
        impf = difference_of_gaussians(imphot, parameters['peakfilter'])
        maxima = maximumfindcall(impf)  # find local maxima
        indmgood = maxima[:, 2] > parameters['peakcutoff']  # filter out maxima below cutoff
        # filter ROI that are too close to inital image edges
        indmgood &= (maxima[:, 0] > parameters['dx']) & (maxima[:, 0] <= size_img[1] - parameters['dx'])
        indmgood &= (maxima[:, 1] > parameters['dx']) & (maxima[:, 1] <= size_img[0] - parameters['dx'])
        maxgood = maxima[indmgood, :]
        
        # Live display of the detected ROIs
        if preview:
            plt.clf()
            plt.imshow(imphot, cmap='gray')
            plt.scatter(maxgood[:, 0], maxgood[:, 1], color='red', s=10, marker='o', label="Detected ROIs")
            plt.title(f"Frame {frame_index} - Detected ROIs")
            plt.legend()
            plt.pause(0.5)
            
        # Extraction and saving of the ROIs
        for k in range(maxgood.shape[0]):
            x_coord, y_coord, intensity = maxgood[k]  # x, y, ...
            # Check if the ROI is within the image
            if (x_coord > parameters['dx'] and x_coord <= size_img[1] - parameters['dx'] and
                y_coord > parameters['dx'] and y_coord <= size_img[0] - parameters['dx']):

                # ROI around the point
                x_min = int(x_coord - parameters['dx'])
                x_max = int(x_coord + parameters['dx'] + 1)
                y_min = int(y_coord - parameters['dx'])
                y_max = int(y_coord + parameters['dx'] + 1)
                sub_img = imphot[y_min:y_max, x_min:x_max]
                if sub_img.shape == (2*parameters['dx'] + 1, 2*parameters['dx'] + 1):
                    subimages.append(sub_img)
                    peakcoords.append([x_coord, y_coord, frame_index])
    if preview:
        plt.ioff()  # Deactivate interactive mode
        plt.show()

    if len(subimages) == 0:
        print("No localization found.")
        return

    # Stack subimages along Z axis => shape = (roifit, roifit, N)
    img_stack = np.stack(subimages, axis=2).astype(np.float32)
    peak_coordinates = np.array(peakcoords, dtype=np.float32)

    # varstack = 0 if not sCMOS camera
    varstack = 0
    
    # Spline fitter call
    resultsh = fit_spline(img_stack, peak_coordinates, parameters, varstack)

    # Saving results
    save_results_python(resultsh, parameters, output_filename=parameters["outputfile"])
    save_as_locpalmtracer(img.shape, resultsh, parameters, parameters['outputfile'])
    print("Fitting done.")
