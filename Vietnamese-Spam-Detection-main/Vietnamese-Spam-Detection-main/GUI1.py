import sys
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QPushButton, QDialogButtonBox, QLabel, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit, QFileDialog 
from PyQt6.QtGui import QPalette, QColor, QIcon, QPixmap, QPainter, QPolygonF 
from PyQt6.QtCore import QSize, Qt, QPointF  
import pandas as pd  
import numpy as np
from pathlib import Path
from keras.models import load_model

class ShowResultWindow(QWidget):
    def __init__(self, message):
        super().__init__()

        self.setWindowTitle("Prediction")
        self.setGeometry(600, 350, 300, 100)
        label = QLabel(message, self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.Input_area = QWidget(self)
        self.Input_area.setGeometry(0, 0, 300, 100)
        self.InputLayout = QVBoxLayout()
        self.InputLayout.addWidget(label)
        self.InputLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.Input_area.setLayout(self.InputLayout)

class AnotherWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lung cancer detector")
        self.setFixedSize(QSize(800, 700))
        self.setStyleSheet('background-color: #CCE5FF;')

        #----------------------------App Start-----------------------------

        self.Input_area = QWidget(self)
        self.Input_area.setGeometry(300, 100, 350, 550)
        self.InputLayout = QVBoxLayout()

        self.Input_add_button = QPushButton("Add image")
        self.Input_add_button.setCheckable(True)
        self.Input_add_button.setFixedSize(150, 40)
        self.Input_add_button.setStyleSheet('background-color: ')
        self.Input_add_button.clicked.connect(self.get_image_from_file)

        self.Input_predict_button = QPushButton("Predict")
        self.Input_predict_button.setCheckable(True)
        self.Input_predict_button.setFixedSize(150, 40)
        self.Input_predict_button.clicked.connect(self.predict)

        self.InputLayout.addWidget(self.Input_add_button)
        self.InputLayout.addWidget(self.Input_predict_button)
        self.InputLayout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.Input_area.setLayout(self.InputLayout)
        self.Input_area.show()
        
        #----------------------------App End------------------------------

        self.filenames = []
        self.predicted = False

        # Load models once when the window is initialized
        self.model1 = load_model('models/imageclassifier1.keras')
        self.model2 = load_model('models/imageclassifier2.keras')
        self.model3 = load_model('models/imageclassifier3.keras')

    def get_image_from_file(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(str(Path('./')))
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            self.filenames = dialog.selectedFiles()
            if self.filenames:
                for filename in self.filenames:
                    pixmap = QPixmap(filename)
                    if not pixmap.isNull():
						# Display the selected image on the main window
                        image_label = QLabel(self)
                        image_label.setPixmap(pixmap)
                        
                        image_label.setScaledContents(True)

                        image_label.setGeometry(170, 80, 450, 450)

                        image_label.show()
						# Add the image to the input list
                    else:
                        print("Error: Failed to load image:", filename)
        return

    def predict(self):
        self.predicted = True

        # Call make_predictions() to generate predictions
        self.infor_outputs = []  # Initialize an empty list
        for filename in self.filenames:
            index = self.make_predictions(filename)
            self.infor_outputs.append(index)

        assert len(self.filenames) == len(self.infor_outputs)

         # Now display the results
        for i in range(len(self.filenames)):
            index = self.infor_outputs[i]
            if index == 0:
                self.open_new_window("The image is predicted to be Benign.")
            elif index == 1:
                self.open_new_window("The image is predicted to be Malignant.")
            elif index == 2:
                self.open_new_window("The image is predicted to be Normal.")
            else:
                self.open_new_window("No validation")

    def open_new_window(self, message):
        self.new_window = ShowResultWindow(message)
        self.new_window.show()

    def make_predictions(self, image_path):
        # Preprocess image
        try:
            img = load_img(image_path, target_size=(256, 256))
            img_array = img_to_array(img)
            scaled_img_array = img_array / 255.0
        except FileNotFoundError:
            print(f"Error: File not found: {image_path}")
            return None  # Hoặc xử lý lỗi theo cách khác

        # Make predictions (tối ưu hóa bằng vectorization)
        predictions1 = self.model1.predict(np.array([scaled_img_array]))
        predictions2 = self.model2.predict(np.array([scaled_img_array]))
        predictions3 = self.model3.predict(np.array([scaled_img_array]))

        # Calculate scores (using NumPy vectorization)
        RC1 = 2 * (1 - np.power(2, predictions1[0] - 1))
        RC2 = 2 * (1 - np.power(2, predictions2[0] - 1))
        RC3 = 2 * (1 - np.power(2, predictions3[0] - 1))
        FRS = RC1 + RC2 + RC3
        CCFS = 1 - (1/3 * (predictions1[0] + predictions2[0] + predictions3[0]))
        FDS = FRS * CCFS

        # Find the index of the highest value in FDS
        min_index = np.argmin(FDS)  # Use np.argmin() for finding the minimum index

        return min_index

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
    
        self.setWindowTitle("Lung cancer detector")
        self.setFixedSize(QSize(800, 700))

        # creating label
        self.label = QLabel(self)
         
        # loading image
        self.pixmap = QPixmap('lung.png')
 
        # adding image to label
        self.label.setPixmap(self.pixmap)
 
        # Optional, resize label to image size
        self.label.resize(800,700)

        self.Main = QWidget(self)
        self.Main.setGeometry(300, 100, 350, 550)
        self.Main_Layout = QVBoxLayout()

        self.button = QPushButton("Start")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.show_new_window)
        self.button.setFixedSize(150,40)
        self.Main_Layout.addWidget(self.button)
        self.Main_Layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
    
        self.Main.setLayout(self.Main_Layout)
        self.Main.show()
    
    def show_new_window(self, checked):
        self.w = AnotherWindow()
        self.w.show()


app = QApplication(sys.argv)
#app.setStyleSheet(stylesheet)

window = AnotherWindow()
window.show()

app.exec()