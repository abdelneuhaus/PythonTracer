import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects

def maximumfindcall(imin):
    """
    Trouve les maxima locaux dans une image 2D en utilisant une approche similaire à l'algorithme de Neubeck et Gool.
    :param imin: Image d'entrée sous forme de tableau NumPy.
    :return: Tableau des maxima sous forme de coordonnées [x, y, intensité].
    """
    # Appliquer un filtre maximum pour trouver les maxima locaux
    neighborhood_size = 3
    data_max = maximum_filter(imin, neighborhood_size)
    maxima = (imin == data_max)
    
    # Supprimer les bords de l'image pour éviter les artefacts
    maxima[0, :] = maxima[-1, :] = maxima[:, 0] = maxima[:, -1] = False
    
    # Trouver les coordonnées des maxima
    labeled, num_objects = label(maxima)
    slices = find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) // 2
        y_center = (dy.start + dy.stop - 1) // 2
        x.append(x_center)
        y.append(y_center)
    
    # Extraire les valeurs des maxima
    intensities = imin[y, x]
    
    # Créer un tableau des maxima sous forme de coordonnées [x, y, intensité]
    maximaout = np.array(list(zip(x, y, intensities)))
    
    return maximaout