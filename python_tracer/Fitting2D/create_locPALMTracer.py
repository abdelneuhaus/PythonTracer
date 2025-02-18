import pandas as pd
from itertools import groupby


def from_plane_to_index(planes):
    """
    From planes list, it will generates index list
    
    eg: [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5]
    Out [1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2]
    """
    transformed_list = []
    for key, group in groupby(planes):
        group_length = len(list(group))
        transformed_list.extend(range(1, group_length + 1))
    
    return transformed_list



def fill_locPALMTracer_file(image_size, planes, centroids, sigmasX, sigmasY, integrated_intensity):
    # header of locPALMTracer file
    metadata_pt = pd.DataFrame(columns=['Width', 'Height', 'nb_Planes', 'nb_Points', 
                                    'Pixel_Size(um)', 'Frame_Duration(s)', 'Gaussian_Fit', 'Spectral'])
    # columns of locPALMTracer file
    localization_pt = pd.DataFrame(columns=['id', 'Plane', 'Index', 'Channel', 'Integrated_Intensity', 
                                   'CentroidX(px)', 'CentroidY(px)', 'SigmaX(px)', 'SigmaY(px)', 
                                   'Angle(rad)', 'MSE(Gauss)', 'CentroidZ(um)', 'MSE_Z(um)'])
    # stack shape TXY
    if len(image_size) == 3:
        metadata_pt.loc[0,'Width'] = image_size[1]
        metadata_pt.loc[0,'Height'] = image_size[2]
        metadata_pt.loc[0,'nb_Planes'] = image_size[0]
    # single frame
    else:
        metadata_pt.loc[0,'Width'] = image_size[0]
        metadata_pt.loc[0,'Height'] = image_size[1]
        metadata_pt.loc[0,'nb_Planes'] = 1

    # nbPoints is length of centroid & integrated_intensity lists
    metadata_pt.loc[0,'nb_Points'] = len(integrated_intensity)
    # for now, default values
    metadata_pt.loc[0,'Pixel_Size(um)'] = 0.160
    metadata_pt.loc[0,'Frame_Duration(s)'] = 0.050
    metadata_pt.loc[0,'Gaussian_Fit'] = 'ToAdd'
    metadata_pt.loc[0,'Spectral'] = False

    
    # now localizations
    localization_pt['id'] = range(1, len(integrated_intensity) + 1) # id ranging from 1 to nbPoints+1 (start at 1 for id)
    localization_pt['Channel'] = [-1] * len(integrated_intensity)
    localization_pt['Index'] = from_plane_to_index(planes) 
    localization_pt['Plane'] = planes
    localization_pt['Integrated_Intensity'] = integrated_intensity
    localization_pt['CentroidX(px)'] = list(zip(*centroids))[0]
    localization_pt['CentroidY(px)'] = list(zip(*centroids))[1]
    localization_pt['SigmaX(px)'] = sigmasX
    localization_pt['SigmaY(px)'] = sigmasY
    localization_pt['Angle(rad)'] = [0] * len(integrated_intensity)
    localization_pt['MSE(Gauss)'] = [-1] * len(integrated_intensity)
    localization_pt['CentroidZ(um)'] = [-1] * len(integrated_intensity)
    localization_pt['MSE_Z(um)'] = [0] * len(integrated_intensity)

    return metadata_pt, localization_pt