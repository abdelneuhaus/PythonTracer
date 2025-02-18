import ctypes
import numpy as np
import tifffile
import os

from Fitting2D.create_locPALMTracer import fill_locPALMTracer_file

def do_fit_and_loc_file(path_stack, threshold=180.5, size_ROI_fit=7):
    """
    Do fitting and save output in the same repertory as path_stack.
    Input: path_stack (pathway of tif file), threshold, size_ROi_fit (size of crop for fitting) 
    """
    mydll = ctypes.cdll.LoadLibrary("./CPU_PALM.dll")
    c_short_pointer = ctypes.POINTER(ctypes.c_ushort)
    c_double_pointer = ctypes.POINTER(ctypes.c_double)

    stack = tifffile.imread(path_stack)

    # fitting parameters
    # ONLY CHANGE theresholdVal & size for now
    potential_points = 4999 # max points per frame approxi. (to divide by 13 because it creates an array with all 13 parameters for each points)
    waveletNo = ctypes.c_uint(1)
    thresholdVal = ctypes.c_double(threshold)  # threshold
    watershedRatio = ctypes.c_double(0)  # watershed or not
    volMin = ctypes.c_double(4)
    intMin = ctypes.c_double(0)
    gaussFit = ctypes.c_ushort(2)  # gauss fit type
    sigma_gaussfit = ctypes.c_double(1)  # initial value for sigma
    theta_gauss_fit = ctypes.c_double(0)  # initial value for theta
    size = ctypes.c_ushort(size_ROI_fit)  # size of ROI around detected molecule

    # stock centroids & integrated_intensity
    centroids = []
    integrated_intensity = []
    sigmasX = []
    sigmasY = []
    planes = []

    for i in range(stack.shape[0]):
        f = stack[i]
        x = f.shape[1]
        y = f.shape[0]
        
        # create empty array to stock detected loc. of the current frame
        empty_array = np.zeros((potential_points,))
        point_list_array = empty_array.ctypes.data_as(c_double_pointer)

        image_array = f.ctypes.data_as(c_short_pointer)

        # actual fit
        mydll._OpenPALMProcessing(image_array, point_list_array, potential_points, y, x, 
                                waveletNo, thresholdVal, watershedRatio, volMin, intMin, 
                                gaussFit, sigma_gaussfit, sigma_gaussfit, theta_gauss_fit, size)

        t = mydll._PALMProcessing()
        pointNumberFrame = (t / 13) - 1 # number of detected locs on the frame
        mydll._closePALMProcessing()

        result = np.ctypeslib.as_array(point_list_array, shape=(potential_points,))

        # stock centroids and ROIs
        for j in range(0, len(result), 13):
            centroidX = result[j + 4]
            centroidY = result[j + 3]
            intensity = result[j + 5] * 2*np.pi* (result[j] * result[j+1])
            if (centroidX, centroidY) != (0, 0) and (centroidX, centroidY) != (-1, -1):
                centroids.append((centroidX, centroidY))
                integrated_intensity.append(intensity)
                sigmasX.append(result[j])
                sigmasY.append(result[j+1])
                planes.append(i+1)

    # generate data to save
    metadata_to_save, loc_to_save = fill_locPALMTracer_file(stack.shape, planes, centroids, sigmasX, sigmasY, integrated_intensity)

    # save locPALMTracer.txt in repertory of image name
    # os.path.basename = get the last part of path, here image_name.tif
    save_path = "./" + os.path.basename(os.path.normpath(path_stack)).replace("tif","PT") + '/locPALMTracer.txt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as file:
        file.write('\t'.join(metadata_to_save.columns) + '\n')
        file.write(metadata_to_save.to_csv(sep='\t', index=False, header=False))
        file.write('\t'.join(loc_to_save.columns) + '\n')
        file.write(loc_to_save.to_csv(sep='\t', index=False, header=False))