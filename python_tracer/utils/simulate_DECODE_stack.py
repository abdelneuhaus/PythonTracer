import decode
import decode.utils
import decode.neuralfitter.train.live_engine
import tifffile as tiff
import numpy as np
import torch


def simulate_DECODE_stack(calib_file, 
                          baseline, 
                          e_per_adu, 
                          em_gain, 
                          pixelsize, 
                          qe, 
                          readout, 
                          spurnoise, 
                          background, 
                          img_size,
                          zrange,
                          nFrames,
                          intensity,
                          sd_intensity,
                          emitterdensity,
                          lifetime, 
                          outputname):
    device = 'cuda:0'  # or 'cpu'
    device_ix = 0  # possibly change device index (only for cuda)
    threads = 4  #  number of threads, useful for CPU heavy computation. Change if you know what you are doing.
    worker = 4  # number of workers for data loading. Change only if you know what you are doing.

    torch.set_num_threads(threads)  # set num threads
    decode.utils.param_io.copy_reference_param('')  # saved in current dir
    param = decode.utils.param_io.load_params('./param_friendly.yaml')  # change path if you load custom file

    param.Hardware.device = device
    param.Hardware.device_ix = device_ix
    param.Hardware.device_simulation = device
    param.Hardware.torch_threads = threads
    param.Hardware.num_worker_train = worker

    param.Camera.baseline = baseline
    param.Camera.e_per_adu = e_per_adu
    param.Camera.em_gain = em_gain
    param.Camera.px_size =[pixelsize, pixelsize] # Pixel Size in nano meter
    param.Camera.qe = qe              # Quantum efficiency
    param.Camera.read_sigma = readout
    param.Camera.spur_noise = spurnoise
    param.Camera.to_dict()

    param.Simulation.bg_uniform = background          # background range to sample from. You can also specify a const. value as 'bg_uniform = 100'
    param.Simulation.emitter_av = emitterdensity    
    param.Simulation.emitter_extent[0] = [-0.5, img_size-0.5]    # Volume in which emitters are sampled. x,y values should not be changed. z-range (in nm) should be adjusted according to the PSF
    param.Simulation.emitter_extent[1] = [-0.5, img_size-0.5]    # Volume in which emitters are sampled. x,y values should not be changed. z-range (in nm) should be adjusted according to the PSF
    param.Simulation.emitter_extent[2] = [-zrange/2, zrange/2]    # Volume in which emitters are sampled. x,y values should not be changed. z-range (in nm) should be adjusted according to the PSF
    param.Simulation.psf_extent[0] = [-0.5, img_size-0.5]
    param.Simulation.psf_extent[1] = [-0.5, img_size-0.5]
    param.Simulation.img_size = [img_size, img_size]
    param.Simulation.intensity_mu_sig = [intensity, sd_intensity]  # Average intensity and its standard deviation
    param.Simulation.lifetime_avg = lifetime                     # Average lifetime of each emitter in frames. A value between 1 and 2 works for most experiments
    param.Simulation.to_dict()

    param.InOut.calibration_file = calib_file
    param.InOut.experiment_out = ''
    param.InOut.to_dict()

    simulator, sim_test = decode.neuralfitter.train.live_engine.setup_random_simulation(param)
    camera = decode.simulation.camera.Photon2Camera.parse(param)

    param = decode.utils.param_io.autoset_scaling(param)

    tar_em, sim_frames, bg_frames = simulator.sample()
    sim_frames = sim_frames.cpu()
    sim_frames_16bits = sim_frames.numpy()[:nFrames,:,:].astype(np.uint16)
    output = outputname+".tif"
    tiff.imwrite(output, sim_frames_16bits, dtype=np.uint16)