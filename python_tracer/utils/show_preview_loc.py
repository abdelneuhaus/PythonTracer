import ctypes
import tifffile
import numpy as np

def show_preview_loc(frame_idx, path_stack, threshold, size_ROI_fit):
    """
    Show selected points for the fitting process.
    Input: frame_idx (current frame to access)

    Poorly computed: do a fit, draw ROIs and add them
    """

    mydll = ctypes.cdll.LoadLibrary("./CPU_PALM.dll")
    c_short_pointer = ctypes.POINTER(ctypes.c_ushort)
    c_double_pointer = ctypes.POINTER(ctypes.c_double)

    # load stack + check size
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
    size = ctypes.c_ushort(size_ROI_fit)  # size of ROI for the fit (ROIs shown with Preview 2D Localization in PT)

    # initialize size of the crop (size_crop x size_crop)
    # from each coordinates, it will crop a square with the PSF at its center
    size_ROI_crop = size_ROI_fit 

    # stock centroids & ROIs
    centroids = []
    ROIs = []

    
    f = stack[frame_idx]
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
            ROIs.append(ROI)
    
    centroids_np = np.array(centroids)
    
    # Create rectangles for each ROI
    shapes = []
    for centroid in centroids_np:
        x_min = centroid[0] - size_ROI_crop // 2
        x_max = centroid[0] + size_ROI_crop // 2
        y_min = centroid[1] - size_ROI_crop // 2
        y_max = centroid[1] + size_ROI_crop // 2
        rectangle = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])
        shapes.append(rectangle)
    
    return shapes