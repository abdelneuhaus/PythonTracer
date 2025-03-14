import os

def set_parameters(stack_path, 
                   calib_file, 
                   offset=None, 
                   conversion=0.12, 
                   peakfilter=1.2, 
                   peakcutoff=15, 
                   roifit=13, 
                   pixelsize=160):
    """
    Set the parameters for the spline fitter. Some parameters are irrelevant for the spline fitter (raw translation of Matlab code).

    Args:
        stack_path: path to the stack file
        calib_file: path to the calibration file
        offset: offset value
        conversion: conversion value EM
        peakfilter: peak filter size of gaussian filter
        peakcutoff: peak cutoff value
        roifit: ROI size around the peak
        outputfile: output file path (need to be removed)
        pixelsize: pixel size in nm
    Returns: 
        parameters: dictionary containing the parameters.
    """
    if not os.path.isdir("results/"):
        os.makedirs("results/")
    parameters = {}
    parameters["imagefile"] = stack_path
    parameters["calibfile"] = calib_file
    parameters["offset"] = offset
    parameters["conversion"] = conversion
    parameters["peakfilter"] = peakfilter
    parameters["peakcutoff"] = peakcutoff
    parameters["roifit"] = roifit
    parameters["bidirectional"] = 0
    parameters["mirror"] = 0
    parameters["status"] = []
    parameters["outputfile"] = "results/"+os.path.basename(os.path.normpath(parameters['imagefile'])).replace("tif","csv")
    parameters["outputformat"] = 'csv'
    parameters["pixelsize"] = pixelsize
    parameters["loader"] = 1
    parameters["mij"] = []
    parameters["backgroundmode"] = 'Difference of Gaussians (fast)'
    parameters["isscmos"] = 0
    parameters["scmosfile"] = ''
    parameters["preview"] = 0
    return parameters
