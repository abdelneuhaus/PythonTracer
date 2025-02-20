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
    Cette fonction lit un gros stack TIFF en mode lazy via Dask,
    détecte les PSFs (maxima locaux), extrait les ROIs, puis réalise
    un fit spline 3D et sauvegarde les résultats.

    Parameters
    ----------
    parameters : dict
        Dictionnaire contenant les paramètres nécessaires :
        - imagefile : str (chemin vers le TIFF)
        - calibfile : str (chemin vers le .mat de calibration 3D)
        - offset : float (valeur de l'offset caméra)
        - conversion : float (gain, pixels -> photons)
        - peakfilter : float (sigma utilisé pour le Difference of Gaussians)
        - peakcutoff : float (seuil d'intensité pour valider un pic)
        - roifit : int (taille du ROI, ex: 13 => ROI de 13x13)
        - outputfile : str (nom de fichier de sortie)
        - (Optionnel) dx, dz, z0, coeff : si vous avez déjà calibré
    """

  

    # Lecture "lazy" du stack TIFF via dask-image
    # arr aura la forme (Nframes, height, width), mais pas chargé en RAM d'un coup.
    arr = imread(parameters['imagefile'])  
    nframes = arr.shape[0]

    # Calcule dx = rayon du ROI (ex: roifit=13 => dx=6)
    parameters['dx'] = math.floor(parameters['roifit'] / 2)

    # Lecture du fichier de calibration 3D (si indiqué)
    parameters['isspline'] = True
    if parameters.get('calibfile', None):
        try:
            cal = io.loadmat(parameters['calibfile'])
            # On suppose que 'SXY.cspline.dz', 'SXY.cspline.z0', 'SXY.cspline.coeff' existent
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


    # Définition d'une fonction locale pour traiter UNE frame :
    #    - Conversion photons
    #    - Filtre DoG
    #    - Détection maxima
    #    - Extraction sub-ROIs
    def detect_subimages_one_frame(i):
        """
        Détecte et extrait les ROIs de la frame d'indice i.
        Retourne : (subimages, peakcoords) pour cette frame
        """
        # On force l'évaluation lazy -> numpy array en RAM
        frame_data = arr[i].compute()
        size_img = frame_data.shape

        # Conversion offset/gain
        imphot = (frame_data.astype(np.float32) - parameters['offset']) * parameters['conversion']

        # Filtre Difference of Gaussians
        impf = difference_of_gaussians(imphot, parameters['peakfilter'])

        # Recherche de maxima
        maxima = maximumfindcall(impf)

        # Filtre par le seuil d'intensité et par les bords
        dx = parameters['dx']
        indmgood = (
            (maxima[:, 2] > parameters['peakcutoff']) &
            (maxima[:, 0] > dx) & (maxima[:, 0] < size_img[1] - dx) &
            (maxima[:, 1] > dx) & (maxima[:, 1] < size_img[0] - dx)
        )
        maxgood = maxima[indmgood, :]

        # Extraction des ROIs
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

    # Création d'un Dask Bag pour itérer sur toutes les frames [0..nframes-1]
    # On map la fonction detect_subimages_one_frame sur chaque indice de frame.
    bag = db.from_sequence(range(nframes), partition_size=50)
    results = bag.map(detect_subimages_one_frame).compute()  # déclenche la parallélisation

    # On concatène tous les subimages et peakcoords de toutes les frames
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

    # Empilage en un seul tableau 3D : shape = (roifit, roifit, nb_spots)
    img_stack = np.stack(all_subimages, axis=2).astype(np.float32)
    peak_coordinates = np.array(all_peakcoords, dtype=np.float32)

    # Ici, varstack=0 si pas de bruit sCMOS particulier
    varstack = 0

    # Fit spline selon la calibration
    if parameters['isspline']:
        print("Lancement du fit spline 3D.")
    else:
        print("Lancement d'un fit (gaussien par ex.) - pas implémenté ici.")
    
    # On suppose que vous appelez quand même fit_spline (qui gère le mode Gauss si 'isspline' = False ?)
    resultsh = fit_spline(img_stack, peak_coordinates, parameters, varstack)

    # Sauvegarde des résultats
    save_results_python(resultsh, parameters, output_filename=parameters["outputfile"])
    
    # Sauvegarde style locpalmtracer
    save_as_locpalmtracer((arr.shape[1], arr.shape[2]), resultsh, parameters, parameters["outputfile"])

    print("Fit terminé et résultats sauvegardés. Fin de la fonction.")
