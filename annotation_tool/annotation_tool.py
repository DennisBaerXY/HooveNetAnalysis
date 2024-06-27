import os
import random
import pandas as pd
from glob import glob
import shutil
import time
import torch

from PyQt5 import QtCore, QtGui, QtWidgets

from hoovenet.model import HoovesModel
from hoovenet.utils import load_weights, model_predict
from common.constants import FRAME_DIR, LABELED_FRAMES_DIR, ANNOTATIONS_FILE, LABELED_FRAMES_FILE, BEST_MODEL_PATH

# Load existing annotations
annotations = pd.read_csv(ANNOTATIONS_FILE)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HoovesModel().to(device)
load_weights(model)
model.eval()





class HoofAnnotationTool(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.frame_paths = glob(os.path.join(FRAME_DIR, '*.png'))
        random.shuffle(self.frame_paths)
        self.total_frames = len(self.frame_paths)

        self.annotations = pd.read_csv(ANNOTATIONS_FILE)
        self.annotation_state = {'left_back': None, 'right_back': None, 'left_front': None, 'right_front': None}
        self.frames_labeled = 0
        self.start_time = time.time()
        self.total_labeling_time = 0

        self.last_skipped_frame = None
        self.last_saved_annotation = None
        self.skip_confirm = False

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_info_label)
        self.timer.start(1000)  # Update every second

        with open(LABELED_FRAMES_FILE, 'r') as f:
            self.labeled_frames = set(f.read().splitlines())

        print(f"Loaded {self.total_frames} frames.")
        self.process_next_frame()

    def initUI(self):
        self.setWindowTitle('Hoof Annotation Tool')
        self.setGeometry(100, 100, 1200, 800)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(50, 50, 864, 396)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        button_layout = QtWidgets.QHBoxLayout()

        self.hoof_buttons = {}

        self.default_button_color = 'lightgray'
        self.selected_button_color = 'lightgreen'
        hoof_labels = ['left_back', 'right_back', 'left_front', 'right_front']

        key_description = {
            'left_back': ('Z', 'H'),
            'right_back': ('U', 'J'),
            'left_front': ('I', 'K'),
            'right_front': ('O', 'L')
        }

        for hoof in hoof_labels:
            vbox = QtWidgets.QVBoxLayout()
            label = QtWidgets.QLabel(
                f"{hoof.replace('_', ' ').title()} ({key_description[hoof][0]}: On, {key_description[hoof][1]}: Off)")
            vbox.addWidget(label)

            button_yes = QtWidgets.QPushButton('On Ground', self)
            button_yes.setStyleSheet(f"background-color: {self.default_button_color}")
            button_yes.clicked.connect(lambda _, h=hoof: self.update_annotation(h, 1))
            vbox.addWidget(button_yes)

            button_no = QtWidgets.QPushButton('Off Ground', self)
            button_no.setStyleSheet(f"background-color: {self.default_button_color}")
            button_no.clicked.connect(lambda _, h=hoof: self.update_annotation(h, 0))
            vbox.addWidget(button_no)

            self.hoof_buttons[hoof] = {'on_ground': button_yes, 'off_ground': button_no}

            button_layout.addLayout(vbox)

        control_layout = QtWidgets.QHBoxLayout()
        self.skip_button = QtWidgets.QPushButton('Skip Frame (E)', self)
        self.skip_button.clicked.connect(self.skip_frame)
        control_layout.addWidget(self.skip_button)

        self.revert_button = QtWidgets.QPushButton('Revert/Back', self)
        self.revert_button.clicked.connect(self.revert_or_back)
        control_layout.addWidget(self.revert_button)

        self.save_button = QtWidgets.QPushButton('Save Annotation (W)', self)
        self.save_button.clicked.connect(self.save_annotations)
        control_layout.addWidget(self.save_button)

        self.quit_button = QtWidgets.QPushButton('Quit (Q)', self)
        self.quit_button.clicked.connect(QtCore.QCoreApplication.instance().quit)
        control_layout.addWidget(self.quit_button)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(control_layout)

        self.info_label = QtWidgets.QLabel(self)
        self.info_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
        self.info_label.setFixedHeight(100)
        self.info_label.setFixedWidth(300)
        main_layout.addWidget(self.info_label)

        self.key_info_label = QtWidgets.QLabel(self)
        self.key_info_label.setText(
            "Key Bindings:\n"
            "Z/U/I/O: On Ground (Left Back/Right Back/Left Front/Right Front)\n"
            "H/J/K/L: Off Ground (Left Back/Right Back/Left Front/Right Front)\n"
            "E: Skip Frame\n"
            "W: Save Annotation\n"
            "Q: Quit"
        )
        self.key_info_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.key_info_label.setFixedHeight(100)
        self.key_info_label.setFixedWidth(300)
        main_layout.addWidget(self.key_info_label)

        self.setLayout(main_layout)

        self.keyPressEvent = self.keyPressEventOverride  # Override key press event

    def process_next_frame(self):
        if not self.frame_paths:
            QtWidgets.QMessageBox.information(self, 'Info', 'No more frames to annotate.')
            QtCore.QCoreApplication.instance().quit()
            return

        self.frame_path = self.frame_paths.pop()
        self.frame_name = os.path.basename(self.frame_path)

        if self.frame_name in self.labeled_frames:
            self.process_next_frame()
            return

        img = QtGui.QImage(self.frame_path)
        if img.isNull():
            print(f"Error loading frame: {self.frame_path}")
            self.process_next_frame()
            return

        pixmap = QtGui.QPixmap.fromImage(img)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio,
                               QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        print(f"Loaded frame: {self.frame_path}")

        # Use model predictions for initial state
        predictions = model_predict(self.frame_path, model, device)
        self.annotation_state = predictions

        print(f"Model predictions: {predictions} for frame {self.frame_name} (On Ground: 1, Off Ground: 0)")

        for hoof, prediction in predictions.items():
            if prediction == 1:
                self.hoof_buttons[hoof]['on_ground'].setStyleSheet(f"background-color: {self.selected_button_color}")
                self.hoof_buttons[hoof]['off_ground'].setStyleSheet(f"background-color: {self.default_button_color}")
            else:
                self.hoof_buttons[hoof]['on_ground'].setStyleSheet(f"background-color: {self.default_button_color}")
                self.hoof_buttons[hoof]['off_ground'].setStyleSheet(f"background-color: {self.selected_button_color}")


        self.update_info_label()

    def save_annotations(self):
        if None in self.annotation_state.values():
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Please annotate all hooves before saving.')
            return

        new_annotation = {
            'frame': self.frame_name,
            'left_back': self.annotation_state['left_back'],
            'right_back': self.annotation_state['right_back'],
            'left_front': self.annotation_state['left_front'],
            'right_front': self.annotation_state['right_front']
        }

        annotations_df = pd.read_csv(ANNOTATIONS_FILE)
        new_annotation_df = pd.DataFrame([new_annotation])
        annotations_df = pd.concat([annotations_df, new_annotation_df], ignore_index=True)
        annotations_df.to_csv(ANNOTATIONS_FILE, index=False)

        labeled_frame_path = os.path.join(LABELED_FRAMES_DIR, self.frame_name)
        shutil.copy(self.frame_path, labeled_frame_path)

        self.last_saved_annotation = new_annotation  # Store the last saved annotation
        self.frames_labeled += 1
        self.total_labeling_time += time.time() - self.start_time
        self.start_time = time.time()

        with open(LABELED_FRAMES_FILE, 'a') as f:
            f.write(f"{self.frame_name}\n")

        self.labeled_frames.add(self.frame_name)

        print(f"Annotations saved for {self.frame_name} and frame copied to labeled_frames.")
        self.process_next_frame()

    def skip_frame(self):
        if not self.skip_confirm:
            self.skip_confirm = True
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Press Skip again to confirm.')
            return
        else:
            self.last_skipped_frame = self.frame_path
            print(f"Frame {self.frame_name} skipped.")
            self.skip_confirm = False
            self.process_next_frame()

    def revert_or_back(self):
        if self.last_skipped_frame:
            frame_name = os.path.basename(self.last_skipped_frame)
            self.frame_paths.append(self.last_skipped_frame)
            self.last_skipped_frame = None
            QtWidgets.QMessageBox.information(self, 'Info', 'Last skipped frame reverted.')
            random.shuffle(self.frame_paths)  # Re-shuffle to ensure reverted frame is not the next frame
            self.process_next_frame()
        elif self.last_saved_annotation:
            # Remove the last saved annotation from the CSV
            annotations_df = pd.read_csv(ANNOTATIONS_FILE)
            annotations_df = annotations_df[annotations_df['frame'] != self.last_saved_annotation['frame']]
            annotations_df.to_csv(ANNOTATIONS_FILE, index=False)

            frame_name = self.last_saved_annotation['frame']
            self.labeled_frames.remove(frame_name)

            with open(LABELED_FRAMES_FILE, 'w') as f:
                f.write("\n".join(self.labeled_frames) + "\n")

            self.last_saved_annotation = None
            QtWidgets.QMessageBox.information(self, 'Info', 'Last saved annotation reverted.')
            random.shuffle(self.frame_paths)  # Re-shuffle to ensure reverted frame is not the next frame
            self.process_next_frame()
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No frame to revert or go back to.')

    def update_annotation(self, hoof, state):
        self.annotation_state[hoof] = state
        self.hoof_buttons[hoof]['on_ground'].setStyleSheet(
            f"background-color: {self.selected_button_color if state == 1 else self.default_button_color}")
        self.hoof_buttons[hoof]['off_ground'].setStyleSheet(
            f"background-color: {self.selected_button_color if state == 0 else self.default_button_color}")

    def keyPressEventOverride(self, event):
        key = event.key()

        key_map = {
            QtCore.Qt.Key_Z: ('left_back', 1),
            QtCore.Qt.Key_U: ('right_back', 1),
            QtCore.Qt.Key_I: ('left_front', 1),
            QtCore.Qt.Key_O: ('right_front', 1),
            QtCore.Qt.Key_H: ('left_back', 0),
            QtCore.Qt.Key_J: ('right_back', 0),
            QtCore.Qt.Key_K: ('left_front', 0),
            QtCore.Qt.Key_L: ('right_front', 0),
            QtCore.Qt.Key_E: self.skip_frame,
            QtCore.Qt.Key_W: self.save_annotations,
            QtCore.Qt.Key_Q: QtCore.QCoreApplication.instance().quit
        }

        if key in key_map:
            action = key_map[key]
            if isinstance(action, tuple):
                hoof, state = action
                self.update_annotation(hoof, state)
            elif callable(action):
                action()

    def update_info_label(self):
        total_labeled = len(self.annotations)
        total_left = self.total_frames - total_labeled
        elapsed_time = time.time() - self.start_time
        avg_time_per_frame = self.total_labeling_time / self.frames_labeled if self.frames_labeled > 0 else 0

        total_seconds = int(elapsed_time)
        minutes = total_seconds // 60
        seconds = total_seconds % 60

        if minutes > 0:
            elapsed_time_formatted = f"{minutes}:{seconds:02d} Min"
        else:
            elapsed_time_formatted = f"{seconds} Seconds"

        avg_seconds = int(avg_time_per_frame)
        avg_minutes = avg_seconds // 60
        avg_seconds = avg_seconds % 60

        if avg_minutes > 0:
            avg_time_formatted = f"{avg_minutes}:{avg_seconds:02d} Min"
        else:
            avg_time_formatted = f"{avg_seconds} Seconds"

        info_text = (f"Total Frames Left: {total_left}\n"
                     f"Total Frames Labeled: {total_labeled}\n"
                     f"Frames Labeled (this session): {self.frames_labeled}\n"
                     f"Total Time Elapsed: {elapsed_time_formatted}\n"
                     f"Avg Time per Frame: {avg_time_formatted}")

        self.info_label.setText(info_text)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ex = HoofAnnotationTool()
    ex.show()
    app.exec_()
