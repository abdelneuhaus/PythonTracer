from PyQt5.QtWidgets import QCheckBox, QVBoxLayout, QWidget
from .show_preview_loc import show_preview_loc

class PreviewCheckButton(QWidget):
    def __init__(self, viewer, get_path_stack, get_threshold, get_roi_fit):
        super().__init__()
        self.viewer = viewer
        self.get_path_stack = get_path_stack
        self.get_threshold = get_threshold
        self.get_roi_fit = get_roi_fit
        self.initUI()
        self.viewer.dims.events.current_step.connect(self.on_frame_change)

    def initUI(self):
        self.checkbox = QCheckBox('Show ROIs', self)
        self.checkbox.stateChanged.connect(self.update_rois)
        layout = QVBoxLayout()
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def update_rois(self):
        if self.checkbox.isChecked():
            self.show_rois()
        else:
            self.hide_rois()

    def show_rois(self):
        path_stack = self.get_path_stack()
        if path_stack:
            frame_idx = self.viewer.dims.current_step[0]
            shapes = show_preview_loc(frame_idx, path_stack, self.get_threshold(), self.get_roi_fit())
            if 'ROIs' in self.viewer.layers:
                self.viewer.layers['ROIs'].data = shapes
            else:
                self.viewer.add_shapes(shapes, shape_type='rectangle', edge_color='green', face_color='transparent', name='ROIs')

    def hide_rois(self):
        if 'ROIs' in self.viewer.layers:
            self.viewer.layers.remove('ROIs')

    def on_frame_change(self, event):
        if self.checkbox.isChecked():
            self.show_rois()  # Show ROIs for the new frame