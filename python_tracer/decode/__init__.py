# """
# DECODE
# This software package implements a DeepLearning based framework for high-density fitting in SMLM.

# """

__version__ = '0.10.2'  # do not modify by hand set and sync with bumpversion
__author__ = 'Lucas-Raphael Mueller, Artur Speiser'
__repo__ = 'https://github.com/TuragaLab/DECODE/master/gateway.yaml'  # main repo
__gateway__ = 'https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml'  # gateway

import warnings

import python_tracer.decode.evaluation 
import python_tracer.decode.generic
import python_tracer.decode.neuralfitter
import python_tracer.decode.plot
import python_tracer.decode.renderer
import python_tracer.decode.simulation
from python_tracer.decode.generic.emitter import EmitterSet, RandomEmitterSet, CoordinateOnlyEmitter

# check device capability
import torch
import python_tracer.decode.utils.hardware as hardware

if torch.cuda.is_available():
    device_capa = hardware.get_device_capability()
    if float(device_capa) < 3.7:
        warnings.warn(
            f"Your GPU {torch.cuda.get_device_name()} has cuda capability {device_capa} and is no longer supported (minimum is 3.7)."
            f"\nIf you have multiple devices make sure to select the index of the most modern one."
            f"\nOtherwise you can use your CPU to run DECODE or switch to Google Colab.", category=UserWarning)
