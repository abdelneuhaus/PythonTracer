import os

import pandas as pd
import numpy as np
from itertools import groupby


def from_plane_to_index(planes):
    """
    This function generates index list from list of plane
    Args:
        Planes: array of int
    Returns:
        transformed_list: list of int
    
    eg: [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5]
    Out [1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2]
    """
    transformed_list = []
    for key, group in groupby(planes):
        group_length = len(list(group))
        transformed_list.extend(range(1, group_length + 1))
    
    return transformed_list


def save_as_locpalmtracer(image_size, results, parameters, outputfile_name='results.txt'):
    """
    This function initializes and fills two dict() compatible with PALMTracer file format
    Args:
        image_size: return of np.shape, (z, y, x) format
        results: int, number of planes (image_size[0])
        parameters: dict(), contains parameters used for spline fitting
        outputfile_name: not used
    Returns:
        None
    """
    # header of locPALMTracer file
    metadata_pt = pd.DataFrame(columns=['Width', 'Height', 'nb_Planes', 'nb_Points', 
                                    'Pixel_Size(um)', 'Frame_Duration(s)', 'Gaussian_Fit', 'Spectral'])

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
    metadata_pt.loc[0,'nb_Points'] = len(results[:,0])
    # for now, default values
    metadata_pt.loc[0,'Pixel_Size(um)'] = parameters['pixelsize']/1000
    metadata_pt.loc[0,'Frame_Duration(s)'] = 0.050
    metadata_pt.loc[0,'Gaussian_Fit'] = 'No_its_Spline_Fit'
    metadata_pt.loc[0,'Spectral'] = False
    

    N = results.shape[0]
    results_extended = np.zeros((N, 16), dtype=results.dtype)
    results_extended[:, :12] = results

    # x, y... values times pixelsize (in nm)
    results_extended[:, [12, 14]] = results_extended[:, [1, 6]] * parameters["pixelsize"]
    results_extended[:, [13, 15]] = results_extended[:, [2, 7]] * parameters["pixelsize"]
    results = results_extended
    
    columns = [
        'frame', 'x_pix', 'y_pix', 'z_nm',
        'photons', 'background',
        'crlb_x', 'crlb_y', 'crlb_z',
        'crlb_photons', 'crlb_background',
        'logLikelyhood',
        'x_nm', 'y_nm',
        'crlb_xnm', 'crlb_ynm']
    resultstable = pd.DataFrame(results, columns=columns)        

    # now localizations
    # columns of locPALMTracer file
    localization_pt = pd.DataFrame(columns=['id', 
                                            'Plane', 
                                            'Intensity_Photons', 
                                            'CentroidX(px)', 
                                            'CentroidY(px)', 
                                            'CRLB_X(px)', 
                                            'CRLB_Y(px)',
                                            'CentroidX(um)',
                                            'CentroidY(um)', 
                                            'CentroidZ(um)', 
                                            'CRLB_X(um)',
                                            'CRLB_Y(um)', 
                                            'CRLB_Z(um)',
                                            'logLikelyhood'])
    
    localization_pt['id'] = range(1, len(list(resultstable['photons'])) + 1) # id ranging from 1 to nbPoints+1 (start at 1 for id)
    localization_pt['Plane'] = [int(i) for i in list(resultstable['frame'])]
    localization_pt['Intensity_Photons'] = list(resultstable['photons'])
    localization_pt['CentroidX(px)'] = list(resultstable['x_pix'])
    localization_pt['CentroidY(px)'] = list(resultstable['y_pix'])
    localization_pt['CRLB_X(px)'] = list(resultstable['crlb_x'])
    localization_pt['CRLB_Y(px)'] = list(resultstable['crlb_y'])
    localization_pt['CentroidX(um)'] = list((resultstable['x_nm']*parameters['pixelsize']/1000))
    localization_pt['CentroidY(um)'] = list((resultstable['y_nm']*parameters['pixelsize']/1000))
    localization_pt['CentroidZ(um)'] = list(resultstable['z_nm']*parameters['pixelsize']/1000)
    localization_pt['CRLB_X(um)'] = list((resultstable['crlb_xnm']*parameters['pixelsize']/1000))
    localization_pt['CRLB_Y(um)'] = list((resultstable['crlb_ynm']*parameters['pixelsize']/1000))
    localization_pt['CRLB_Z(um)'] = list((resultstable['crlb_z']*parameters['pixelsize']/1000))
    localization_pt['logLikelyhood(um)'] = list(resultstable['logLikelyhood'])
    save_path = "./results/" +os.path.basename(os.path.normpath(parameters['imagefile'])).replace("tif","PT")+ "/locPALMTracer.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as file:
        file.write('\t'.join(metadata_pt.columns) + '\n')
        file.write(metadata_pt.to_csv(sep='\t', index=False, header=False))
        file.write('\t'.join(localization_pt.columns) + '\n')
        file.write(localization_pt.to_csv(sep='\t', index=False, header=False))
    print(f"Results also saved in {save_path} in PT-like file")