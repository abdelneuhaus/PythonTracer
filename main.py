import napari
import tifffile
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from magicgui import magicgui
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog, QTabWidget, QWidget, QVBoxLayout

from python_tracer.Fitting2D.do_fit_and_loc_file import do_fit_and_loc_file

# from python_tracer import do_fit_and_loc_file
from python_tracer.Fitting2D.do_fit_and_gallery import do_fit_and_gallery
from python_tracer.Fitting2D.parse_loc import read_locPALMTracer
from python_tracer.Fitting2D.parse_track import read_trcPALMTracer
from python_tracer.SplineFitterSMAP.preview_splinefitter import previews_splinefitter
from python_tracer.SplineFitterSMAP.set_parameters import set_parameters
from python_tracer.SplineFitterSMAP.cspline_fitter import cspline_fitter
# from python_tracer.SplineFitterSMAP.cspline_fitter_dask import cspline_fitter_dask    # test
from python_tracer.utils.simulate_DECODE_stack import simulate_DECODE_stack
from python_tracer.utils.PreviewCheckButton import PreviewCheckButton


path_stack = None
mat_file = None
parameters = None

@magicgui(call_button="Load Stack (.tif only for now)")
def load_image():
    global path_stack
    path_stack, _ = QFileDialog.getOpenFileName(None, "Open TIFF File", "", "TIFF files (*.tif *.tiff)")
    if path_stack:
        stack = tifffile.imread(path_stack)
        stack4d = np.expand_dims(stack, 1)
        show_info(f'Image loaded')
        viewer.layers.clear()
        viewer.add_image(stack4d, name=os.path.basename(os.path.normpath(path_stack)), scale=[1, 1, 1])
        global save_path_fit, save_path_gallery
        save_path_fit = './' + str(os.path.basename(os.path.normpath(path_stack)).replace("tif","PT"))
        save_path_gallery = './' + str(os.path.basename(os.path.normpath(path_stack)).replace(".tif","_")) + "gallery_of_rois.tif"
    return path_stack


@magicgui(call_button="Load Stack (.tif only for now)")
def load_image_simu():
    global path_stack
    path_stack, _ = QFileDialog.getOpenFileName(None, "Open TIFF File", "", "TIFF files (*.tif *.tiff)")
    if path_stack:
        stack = tifffile.imread(path_stack)
        stack4d = np.expand_dims(stack, 1)
        show_info(f'Image loaded')
        viewer.layers.clear()
        viewer.add_image(stack4d, name=os.path.basename(os.path.normpath(path_stack)), scale=[1, 1, 1])
        global save_path_fit, save_path_gallery
        save_path_fit = './' + str(os.path.basename(os.path.normpath(path_stack)).replace("tif","PT"))
        save_path_gallery = './' + str(os.path.basename(os.path.normpath(path_stack)).replace(".tif","_")) + "gallery_of_rois.tif"
    return path_stack


@magicgui(call_button="Launch 2D Fit")
def fit_widget(Threshold=180.5, ROI_fit=7):
    if path_stack:
        do_fit_and_loc_file(path_stack, Threshold, ROI_fit)
        show_info("Fit done.")
        show_info(f'File saved in {save_path_fit}')
    else:
        show_info(f'No image opened. Load an image first please.')


@magicgui(call_button="Launch PSF Gallery Creation")
def gallery_widget(Threshold=180.5, ROI_fit=7, ROI_PSF_Crop=8, Number_Of_ROIs_Per_Line=16):
    if path_stack:
        do_fit_and_gallery(path_stack, Threshold, ROI_fit, ROI_PSF_Crop, Number_Of_ROIs_Per_Line)
        show_info(f'Gallery stack saved in {save_path_gallery}')
    else:
        show_info(f'No image opened. Load an image first please.')


@magicgui(call_button="Load locPALMTracer text file")
def load_locPALMTracer_file():
    locPT_pathway, _ = QFileDialog.getOpenFileName(None, "Open locPALMTracer File", "", "All Files (*)")
    if locPT_pathway:
        data, coordinates = read_locPALMTracer(locPT_pathway)
        circles = np.array([(t, x, y) for t,x,y in coordinates])
        viewer.add_points(circles, size=2, name='Locs', scale=[1, 1, 1])
        show_info(f'Locs loaded.')
    else:
        show_info(f'No file openend. Load a locPALMTracer file first please.')


@magicgui(call_button="Load trcPALMTracer text file")
def load_trcPALMTracer_file():
    trcPT_pathway, _ = QFileDialog.getOpenFileName(None, "Open trcPALMTracer File", "", "All Files (*)")
    if trcPT_pathway:
        tracks = read_trcPALMTracer(trcPT_pathway)
        viewer.add_tracks(tracks, name='Tracks', scale=[1,1,1])
        show_info(f'Tracks loaded.')
    else:
        show_info(f'No file openend. Load a trcPALMTracer file first please.')



# Spline Fitter part
@magicgui(call_button="Load Image to Fit (tif only for now)")
def load_smlm_image():
    global path_stack
    path_stack, _ = QFileDialog.getOpenFileName(None, "Open TIFF File", "", "TIFF files (*.tif *.tiff)")
    show_info(f'Image loaded')

@magicgui(call_button="Load PSF model (MATLAB File)")
def load_mat_file_fit():
    global mat_file
    mat_file, _ = QFileDialog.getOpenFileName(None, "Open MATLAB File", "", "MATLAB files (*.mat)")
    show_info(f'PSF model loaded')


@magicgui(call_button="Load PSF model (MATLAB File)")
def load_mat_file_simu():
    global mat_file
    mat_file, _ = QFileDialog.getOpenFileName(None, "Open MATLAB File", "", "MATLAB files (*.mat)")
    show_info(f'PSF model loaded')


@magicgui(call_button="Preview before fitting")
def preview_before_localization(Offset=498, 
                    Conversion=0.12, 
                    PeakFilter=1.2, 
                    PeakCutoff=15, 
                    ROIFit=13, 
                    PixelSize=160, 
                    Preview_NFrames=1):
    if path_stack and mat_file:
        global parameters
        parameters = set_parameters(path_stack, mat_file, Offset, Conversion, PeakFilter, PeakCutoff, ROIFit, PixelSize)
        img, imphot, maxgood = previews_splinefitter(parameters, Preview_NFrames)
        for f in range(len(img)):
            plt.clf()
            plt.imshow(imphot[f], cmap='gray')
            plt.scatter(maxgood[f][:, 0], maxgood[f][:, 1], color='red', s=10, marker='o', label="Detected ROIs")
            plt.title(f"Frame {f+1} - Detected ROIs")
            plt.legend()
            plt.pause(0.5)
            plt.ioff()  # Deactivate interactive mode
            plt.show()
        plt.close()
    return parameters


@magicgui(call_button="Launch Spline Fitter")
def run_spline_fitter(OutputFile='outputSMAP'):
    global parameters
    output_file = "./"+OutputFile+".csv"
    if path_stack and mat_file:
        if parameters == None:
            show_info(f'Default parameters used. Run Preview once.')
            parameters = set_parameters(path_stack, mat_file, 498, 0.12, 1.2, 15, 13, output_file, 160)
        parameters['outputfile'] = output_file
        start_time = time.time()
        # cspline_fitter_dask(parameters)
        cspline_fitter(parameters)
        show_info(f"Fitting completed in {np.round(time.time() - start_time, 3)} seconds")
    else:
        show_info(f'Load both the stack and the PSF model first please.')



@magicgui(call_button="Z-Colored Vizualisation")
def z_colored_vizualisation():
    output_file, _ = QFileDialog.getOpenFileName(None, "Open CSV File", "", "CSV files (*.csv)")
    if output_file:
        show_info(f'File selected.')
        df = pd.read_csv(
            output_file,
            sep=r'\s+',          # ou sep='\t' si c’est clairement des tabulations
            engine='python',     # parfois utile pour gérer les espaces multiples
            header=0             # indique que la première ligne est l’en-tête
        )

        x = df['x_nm'].values
        y = df['y_nm'].values
        z = df['z_nm'].values
        points_data = np.column_stack((z, y, x))
        norm = plt.Normalize(vmin=z.min(), vmax=z.max())
        colors = plt.cm.viridis(norm(z))  # Nx4 RGBA
        viewer.layers.clear()
        viewer.add_points(
            points_data,
            size=20,
            name='Locs',
            face_color=colors, 
            edge_color='white'
        )
        viewer.dims.ndisplay = 3
    else:
        show_info(f'No file selected.')



@magicgui(call_button="Intensity Colored Vizualisation (some bugs)")
def photons_colored_vizualisation():
    output_file, _ = QFileDialog.getOpenFileName(None, "Open CSV File", "", "CSV files (*.csv)")
    if output_file:
        show_info(f'File selected.')
        df = pd.read_csv(
            output_file,
            sep=r'\s+',          # ou sep='\t' si c’est clairement des tabulations
            engine='python',     # parfois utile pour gérer les espaces multiples
            header=0             # indique que la première ligne est l’en-tête
        )
        x = df['x_nm'].values
        y = df['y_nm'].values
        intensity = df['photons'].values
        points_data = np.column_stack((intensity, y, x))
        norm = plt.Normalize(vmin=intensity.min(), vmax=intensity.max())
        colors = plt.cm.viridis(norm(intensity))  # Nx4 RGBA
        viewer.layers.clear()
        viewer.add_points(
            points_data,
            size=20,
            name='Locs',
            face_color=colors,
            edge_color='white'
        )
        viewer.dims.ndisplay = 3
    else:
        show_info(f'No file selected.')


# Simulation of SMLM stack
@magicgui(call_button="Simulate 3D SMLM Data")
def do_simulation_DECODE(Baseline=498.0,
                        e_per_ADU=3.6,
                        EM_gain=300,
                        Pixelsize_nm=160.0,
                        qe=0.95,
                        Readout_sigma=74.4,
                        Spur_noise=0.005,
                        Background=200,
                        Image_SizeXY=256,
                        Z_Range_nm=2000,
                        Frames_Number=10000,
                        Intensity=13000,
                        Intensity_sd=1000,
                        Avg_Emitter_per_Frames=15,
                        Lifetime=4,
                        Output='3DSMLM_simulation'):
    global mat_file
    if mat_file == None:
        mat_file, _ = QFileDialog.getOpenFileName(None, "Open MATLAB File", "", "MATLAB files (*.mat)")
    simulate_DECODE_stack(mat_file, Baseline, e_per_ADU, EM_gain, Pixelsize_nm, qe, Readout_sigma,
                            Spur_noise, Background, Image_SizeXY, Z_Range_nm, Frames_Number, Intensity,
                            Intensity_sd, Avg_Emitter_per_Frames, Lifetime, Output)
    show_info(f'Simulation and mean PSF saved.')



# Create Napari viewer (main GUI)
viewer = napari.Viewer()
preview_button = PreviewCheckButton(viewer, lambda: path_stack, lambda: fit_widget.Threshold.value, lambda: fit_widget.ROI_fit.value)

# Create a tab widget
tab_widget = QTabWidget()

# Create the first tab and add existing widgets
tab1 = QWidget()
tab1_layout = QVBoxLayout()
tab1_layout.addWidget(load_image.native)
tab1_layout.addWidget(preview_button)
tab1_layout.addWidget(fit_widget.native)
tab1_layout.addWidget(gallery_widget.native)
tab1_layout.addWidget(load_locPALMTracer_file.native)
tab1_layout.addWidget(load_trcPALMTracer_file.native)
tab1.setLayout(tab1_layout)

# Create the second tab for splinefitter smap
tab2 = QWidget()
tab2_layout = QVBoxLayout()
tab2_layout.addWidget(load_smlm_image.native)
tab2_layout.addWidget(load_mat_file_fit.native)
tab2_layout.addWidget(preview_before_localization.native)
tab2_layout.addWidget(run_spline_fitter.native)
tab2_layout.addWidget(z_colored_vizualisation.native)
tab2_layout.addWidget(photons_colored_vizualisation.native)
tab2.setLayout(tab2_layout)


# Create the third tab for simulation with DECODE and uiPSF beads/SMLM PSF extraction
tab3 = QWidget()
tab3_layout = QVBoxLayout()
tab3_layout.addWidget(load_mat_file_simu.native)
tab3_layout.addWidget(do_simulation_DECODE.native)
tab3_layout.addWidget(load_image_simu.native)
tab3.setLayout(tab3_layout)

# Add tabs to the tab widget
tab_widget.addTab(tab1, "Demo PALMTracer")
tab_widget.addTab(tab2, "Spline Fitter")
tab_widget.addTab(tab3, "DECODE Stack Simulation")

# Add the tab widget to the viewer
viewer.window.add_dock_widget(tab_widget, area='right')

if __name__ == '__main__':
    napari.run()