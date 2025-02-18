import python_tracer.decode.simulation
import python_tracer.decode.utils


def setup_random_simulation(param):
    """
        Setup the actual simulation

        0. Define PSF function (load the calibration)
        1. Define our struture from which we sample (random prior in 3D) and its photophysics
        2. Define background and noise
        3. Setup simulation and datasets
        """
    psf = python_tracer.decode.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file).init_spline(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size,
        device=param.Hardware.device_simulation,
        roi_size=param.Simulation.roi_size,
        roi_auto_center=param.Simulation.roi_auto_center
    )

    """Structure Prior"""
    prior_struct = python_tracer.decode.simulation.structure_prior.RandomStructure.parse(param)

    if param.Simulation.mode in ('acquisition', 'apriori'):
        frame_range_train = (0, param.HyperParameter.pseudo_ds_size)

    elif param.Simulation.mode == 'samples':
        frame_range_train = (-((param.HyperParameter.channels_in - 1) // 2),
                             (param.HyperParameter.channels_in - 1) // 2)
    else:
        raise ValueError

    prior_train = python_tracer.decode.simulation.emitter_generator.EmitterSamplerBlinking.parse(
        param, structure=prior_struct, frames=frame_range_train)

    """Define our background and noise model."""
    bg = python_tracer.decode.simulation.background.UniformBackground.parse(param)

    if param.CameraPreset == 'Perfect':
        noise = python_tracer.decode.simulation.camera.PerfectCamera.parse(param)
    elif param.CameraPreset is not None:
        raise NotImplementedError
    else:
        noise = python_tracer.decode.simulation.camera.Photon2Camera.parse(param)

    simulation_train = python_tracer.decode.simulation.simulator.Simulation(psf=psf, em_sampler=prior_train, background=bg,
                                                              noise=noise, frame_range=frame_range_train)

    frame_range_test = (0, param.TestSet.test_size)

    prior_test = python_tracer.decode.simulation.emitter_generator.EmitterSamplerBlinking.parse(
        param, structure=prior_struct, frames=frame_range_test)

    simulation_test = python_tracer.decode.simulation.simulator.Simulation(psf=psf, em_sampler=prior_test, background=bg, noise=noise,
                                                             frame_range=frame_range_test)

    return simulation_train, simulation_test
