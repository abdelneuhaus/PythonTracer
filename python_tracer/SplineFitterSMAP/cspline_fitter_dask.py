import math
import numpy as np
import dask.bag as db
from dask_image.imread import imread
import scipy.io as io

from .difference_of_gaussians import difference_of_gaussians
from .fitspline import fit_spline
from .maximumfindcall import maximumfindcall
from .save_results_python import save_results_python
from .save_as_locpalmtracer import save_as_locpalmtracer


def cspline_fitter_dask(parameters):
    """
    This function reads the stack in lazy mode via Dask. It detects PSFs, extracts ROIs, performs spline fit and saves the results
    Args:
    parameters : dict, parameters required for spline fitting.
    """

    # Lazy reading of stack
    arr = imread(parameters['imagefile'])   # shape is (t, y, x)
    nframes = arr.shape[0]

    # Compute ROI radius
    parameters['dx'] = math.floor(parameters['roifit'] / 2)

    # Read matlab calibration file
    parameters['isspline'] = True
    if parameters.get('calibfile', None):
        try:
            cal = io.loadmat(parameters['calibfile'])
            parameters['dz'] = cal['SXY']['cspline'][0,0]['dz'][0][0][0][0]
            parameters['z0'] = cal['SXY']['cspline'][0,0]['z0'][0][0][0][0]
            coeff = cal['SXY']['cspline'][0,0]['coeff'][0][0][0][0]
            if isinstance(coeff, (list, tuple)):
                coeff = coeff[0]
            parameters['coeff'] = coeff
        except:
            print("Erreur lors du chargement de la calibration, on utilise un fit gaussien ?")
            parameters['isspline'] = False
    else:
        print("Aucun fichier de calibration 3D fourni. On continue quand même.")
        parameters['isspline'] = False


    # Define local function to handle ONE frame (conversion in photon, DoG filter, maxima detection and ROIs extraction)
    def detect_subimages_one_frame(i):
        """
        This function detects and extracts ROI of i-th frame.
        Args:
            i: int, frame index
        Returns:
            subimages: array, ROIs
            peakcoords: array, coordinates
        """
        frame_data = arr[i].compute()
        size_img = frame_data.shape

        # Conversion offset/gain
        imphot = (frame_data.astype(np.float32) - parameters['offset']) * parameters['conversion']

        # Difference of Gaussians
        impf = difference_of_gaussians(imphot, parameters['peakfilter'])

        # Search maxima
        maxima = maximumfindcall(impf)

        # Filtering with threshold and if near a border
        dx = parameters['dx']
        indmgood = (
            (maxima[:, 2] > parameters['peakcutoff']) &
            (maxima[:, 0] > dx) & (maxima[:, 0] < size_img[1] - dx) &
            (maxima[:, 1] > dx) & (maxima[:, 1] < size_img[0] - dx)
        )
        maxgood = maxima[indmgood, :]

        # ROIs extraction
        subimages_frame = []
        peakcoords_frame = []
        for k in range(maxgood.shape[0]):
            x_coord, y_coord, _ = maxgood[k]
            x_min = int(x_coord - dx)
            x_max = int(x_coord + dx + 1)
            y_min = int(y_coord - dx)
            y_max = int(y_coord + dx + 1)

            sub_img = imphot[y_min:y_max, x_min:x_max]
            if sub_img.shape == (2*dx + 1, 2*dx + 1):
                subimages_frame.append(sub_img)
                peakcoords_frame.append([x_coord, y_coord, i+1])  # i+1 => n° de frame (1-based)

        return subimages_frame, peakcoords_frame

    
    # Creation of a Dask bag to iterate over all the frames. Mapping of detect_subimages_one_frames over each index
    bag = db.from_sequence(range(nframes), partition_size=50)
    results = bag.map(detect_subimages_one_frame).compute()  # Actual parallelization

    # Concatenate all ROIs and coordinates
    all_subimages = []
    all_peakcoords = []
    for (subimgs, coords) in results:
        all_subimages.extend(subimgs)
        all_peakcoords.extend(coords)

    nb_spots = len(all_subimages)
    print(f"Détection terminée. Nombre total de ROIs détectées : {nb_spots}")

    if nb_spots == 0:
        print("Aucune localisation détectée. Fin du script.")
        return

    # Stack everything in 3D array (roifit, roifit, nb_detections)
    img_stack = np.stack(all_subimages, axis=2).astype(np.float32)
    peak_coordinates = np.array(all_peakcoords, dtype=np.float32)

    varstack = 0    # not sCMOS camera
    if parameters['isspline']:
        print("Lancement du fit spline 3D.")
    else:
        print("Lancement d'un fit (gaussien par ex.) - pas implémenté ici.")
    
    resultsh = fit_spline(img_stack, peak_coordinates, parameters, varstack)

    save_results_python(resultsh, parameters, output_filename=parameters["outputfile"])
    save_as_locpalmtracer((arr.shape[1], arr.shape[2]), resultsh, parameters, parameters["outputfile"])
    print("Fit terminé et résultats sauvegardés. Fin de la fonction.")
