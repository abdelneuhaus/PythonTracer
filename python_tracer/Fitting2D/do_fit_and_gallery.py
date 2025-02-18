import ctypes
import tifffile as tiff
import os
import numpy as np
# import skimage.io as io

def do_fit_and_gallery(path_stack, threshold=180.0, size_ROI_fit=7, size_ROI_crop=8, number_of_ROI_per_line=5):
    mydll = ctypes.cdll.LoadLibrary("./CPU_PALM.dll")
    c_short_pointer = ctypes.POINTER(ctypes.c_ushort)
    c_double_pointer = ctypes.POINTER(ctypes.c_double)

    # load stack + check size
    stack = tiff.imread(path_stack)

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
    size = ctypes.c_ushort(size_ROI_fit)  # size of ROI for the fit (ROIs shown with Preview 2D Localization in PT)

    # stock centroids & ROIs
    centroids = []
    ROIs = []

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
            if (centroidX, centroidY) != (0, 0) and (centroidX, centroidY) != (-1, -1):
                centroids.append((centroidX, centroidY))
                
                # determine ROIs border (coordinates)
                x_min = max(int(centroidX) - size_ROI_crop // 2, 0)
                x_max = min(int(centroidX) + size_ROI_crop // 2, x)
                y_min = max(int(centroidY) - size_ROI_crop // 2, 0)
                y_max = min(int(centroidY) + size_ROI_crop // 2, y)
                
                # generate ROIs of locs of a single frame
                ROI = f[y_min:y_max, x_min:x_max]
                
                # "fill" ROI if at the border (but create black space, maybe use small value for size_crop)
                if ROI.shape[0] != size_ROI_crop or ROI.shape[1] != size_ROI_crop:
                    padded_ROI = np.zeros((size_ROI_crop, size_ROI_crop), dtype=ROI.dtype)
                    padded_ROI[:ROI.shape[0], :ROI.shape[1]] = ROI
                    ROI = padded_ROI                
                ROIs.append(ROI)

    # gallery_size = size of one ROI * number of ROI per line (32 here)
    # create a stack of size (gallery_size * gallery_size)
    gallery_size = size_ROI_crop*number_of_ROI_per_line
    num_rois = len(ROIs)
    if num_rois == 0:
        print("No ROIs found.")
        gallery_of_rois = np.zeros((gallery_size, gallery_size, 0), dtype=stack.dtype)
    else:
        # determine number of ROI to fill stacks
        num_rois_per_layer = (gallery_size // size_ROI_crop) ** 2
        num_layers = (num_rois + num_rois_per_layer - 1) // num_rois_per_layer  # calculate number of stacks
        
        # create stack
        gallery_of_rois = np.zeros((gallery_size, gallery_size, num_layers), dtype=stack.dtype)
        
        # fill stack
        for idx, ROI in enumerate(ROIs):
            layer = idx // num_rois_per_layer
            position = idx % num_rois_per_layer
            row = (position // (gallery_size // size_ROI_crop)) * size_ROI_crop
            col = (position % (gallery_size // size_ROI_crop)) * size_ROI_crop
            gallery_of_rois[row:row+size_ROI_crop, col:col+size_ROI_crop, layer] = ROI
        
        # if last stack is not full, duplicate last ROIs until it is
        while idx + 1 < num_layers * num_rois_per_layer:
            idx += 1
            layer = idx // num_rois_per_layer
            position = idx % num_rois_per_layer
            row = (position // (gallery_size // size_ROI_crop)) * size_ROI_crop
            col = (position % (gallery_size // size_ROI_crop)) * size_ROI_crop
            gallery_of_rois[row:row+size_ROI_crop, col:col+size_ROI_crop, layer] = ROIs[-1]

    # change shape of stack to have (z, x, y)
    gallery_of_rois_transposed = np.transpose(gallery_of_rois, (2, 0, 1))

    # save TIFF file
    output_filename = "gallery_of_rois_" + os.path.basename(os.path.normpath(path_stack))
    # io.imsave(output_filename, gallery_of_rois_transposed, check_contrast=False)
    tiff.imwrite(output_filename, gallery_of_rois_transposed, dtype=np.uint16)