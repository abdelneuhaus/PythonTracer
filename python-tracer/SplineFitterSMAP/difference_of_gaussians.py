import numpy as np
from SplineFitterSMAP.filter2 import filter2
from SplineFitterSMAP.fspecial import gaussian_filter  # la même que votre code

def difference_of_gaussians(imphot, peakfilter):
    """
    Calcule la Difference of Gaussians (DoG) d'une image.
    Équivalent à:
      hdog = fspecial('gaussian', rsize, p.peakfilter)
           - fspecial('gaussian', rsize, max(1,2.5*p.peakfilter))
      impf = filter2(hdog, (imphot - min(imphot(:,1))))
    """

    # 1) Déterminer la taille du noyau (rsize)
    #    Comme en MATLAB : rsize = 2 * ceil(3 * sigma) + 1
    rsize = 2 * int(np.ceil(3 * peakfilter)) + 1

    # 2) Construire deux noyaux gaussiens
    sigma1 = peakfilter
    sigma2 = max(1, 2.5 * peakfilter)

    gauss1 = gaussian_filter(rsize, sigma1)  # 2D gauss
    gauss2 = gaussian_filter(rsize, sigma2)

    # 3) Difference of Gaussians
    hdog = gauss1 - gauss2

    # 4) Soustraire la valeur minimale de la colonne 1 (MATLAB) => en Python ([:, 0])
    #    Attention: le code MATLAB imphot(:,1) fait référence à TOUTES les lignes, 
    #               colonne 1 (indexée à 1). Donc en Python c'est imphot[:, 0]
    #    S'il s'agit vraiment de la première colonne, on fait:
    dog_input = imphot - np.min(imphot[:, 0])

    # 5) Appliquer la convolution 2D
    impf = filter2(hdog, dog_input, shape='same')

    return impf
