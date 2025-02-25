import ctypes
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit  # initialise CUDA, crée un contexte par défaut. Ne pas supprimer.
from napari.utils.notifications import show_info

# Define constants from definitions.h
BSZ = 64
NK = 128
NV_PS = 5

class Dim3(ctypes.Structure):
    """
    Structure to define a 3D dimension for CUDA kernel launch. 
    Uses for kernel launch by setting DimBlock and DimGrid.
    """
    _fields_ = [
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    ]


def mleFit_LM(imstack, iterations, fitpar, varmap, zstart):
    """ 
    MLE fit using GPU adapted from https://github.com/jries/SMAP/blob/master/fit3Dcspline/source/GPUmleFit_LM_SingleChannel/GPUmleFit_LM/mexFunction.cpp
    Args:
        imstack (np.ndarray): 3D image stack
        iterations (int): number of iterations
        fitpar (np.ndarray): parameters
        varmap (np.ndarray): variance map if sCMOS, else not used
        zstart (float): starting z (depth) value
    Returns:
        Parameters_host (np.ndarray): parameters
        CRLBs_host (np.ndarray): CRLBs
        LogLike_host (np.ndarray): Log likelihood
    
    DLL compilation in Admin VISUAL STUDIO terminal:
    nvcc -ccbin "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64" -shared -o GPUmleFit.dll wrapper.cu GPUmleFit_LM_EMCCD.cu GPUmleFit_LM_sCMOS.cu -Xcompiler "/MD" -Xlinker "/NODEFAULTLIB:LIBCMT"
    Requires Visual Studio 22 installed
    """
    
    spline_xsize, spline_ysize, spline_zsize = fitpar.shape[:3]
    datasize = imstack.shape
    sz = datasize[0]
    
    # Allocate GPU memory for imstack (ROIs) and fitpar (PSF model)
    imstack = np.asfortranarray(imstack)
    imstack_device = cuda.mem_alloc(imstack.nbytes) # objet PyCUDA gérant la mémoire sur GPU
    cuda.memcpy_htod(imstack_device, imstack)
    imstack_device_ptr = ctypes.c_void_p(int(imstack_device))

    fitpar = np.asfortranarray(fitpar)
    fitpar_device = cuda.mem_alloc(fitpar.nbytes)
    cuda.memcpy_htod(fitpar_device, fitpar)
    fitpar_device_ptr = ctypes.c_void_p(int(fitpar_device))


    # GPU buffers for results
    Nfitraw = imstack.shape[2]
    NV_PS = 5
    Parameters_dev = cuda.mem_alloc((Nfitraw*(NV_PS+1))*4)  # float32 => 4 octets
    CRLBs_dev      = cuda.mem_alloc((Nfitraw*NV_PS)*4)
    LogLike_dev    = cuda.mem_alloc(Nfitraw*4)
    Parameters_ptr = ctypes.c_void_p(int(Parameters_dev))
    CRLBs_ptr      = ctypes.c_void_p(int(CRLBs_dev))
    LogLike_ptr    = ctypes.c_void_p(int(LogLike_dev))

    # DLL path
    mydll = ctypes.cdll.LoadLibrary("./python_tracer/utils/GPUmleFit.dll")

    # Wrapper function for the kernel
    mydll.kernel_splineMLEFit_z_EMCCD_wrapper.argtypes = [
        Dim3,
        Dim3,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_int
    ]
    mydll.kernel_splineMLEFit_z_EMCCD_wrapper.restype = None


    # Initialize the grid and block dimensions 
    dimGrid = Dim3(((Nfitraw + BSZ - 1) // BSZ), 1, 1)    
    dimBlock = Dim3(64, 1, 1)

    print("Fitting with cspline")
    start_time = time.time()
    try:
        mydll.kernel_splineMLEFit_z_EMCCD_wrapper(
            dimGrid,
            dimBlock,
            imstack_device_ptr,
            fitpar_device_ptr,
            ctypes.c_int(spline_xsize),
            ctypes.c_int(spline_ysize),
            ctypes.c_int(spline_zsize),
            ctypes.c_int(sz),
            ctypes.c_int(iterations),
            Parameters_ptr,
            CRLBs_ptr,
            LogLike_ptr,
            ctypes.c_float(zstart),   # zstart = float Python
            ctypes.c_int(Nfitraw))
    except Exception as e:
        print("Error during function call:", e)

        
    print(f"Fitting completed in {time.time() - start_time} seconds")
    
    # GPU to CPU
    Parameters_host = np.empty((Nfitraw, NV_PS+1), dtype=np.float32, order='F')
    CRLBs_host = np.empty((Nfitraw, NV_PS), dtype=np.float32, order='F')
    LogLike_host = np.empty((Nfitraw,), dtype=np.float32, order='F')
    cuda.memcpy_dtoh(Parameters_host, Parameters_dev)
    cuda.memcpy_dtoh(CRLBs_host, CRLBs_dev)
    cuda.memcpy_dtoh(LogLike_host, LogLike_dev)

    show_info(f"Number of localizations: {Parameters_host.shape[0]}")

    # Free GPU memory
    imstack_device.free()
    fitpar_device.free()
    Parameters_dev.free()
    CRLBs_dev.free()
    LogLike_dev.free()

    return Parameters_host, CRLBs_host, LogLike_host