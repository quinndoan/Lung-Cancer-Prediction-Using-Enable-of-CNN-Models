import sys
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QSize, Qt

class ShowResultWindow(QWidget):
    def __init__(self, messages):
        super().__init__()
        self.setWindowTitle("Prediction Results")
        self.setGeometry(600, 350, 400, 350)
        self.setStyleSheet("background-color: #F0F8FF;")
        
        layout = QVBoxLayout()
        
        header_label = QLabel("Lung Cancer Prediction Results", self)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4682B4;")
        layout.addWidget(header_label)
        
        for message in messages:
            message_label = QLabel(message, self)
            message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message_label.setStyleSheet("font-size: 14px; color: #333333;")
            layout.addWidget(message_label)
        
        explanation_label = QLabel(
            "<b>Explanation:</b><br>"
            "Benign: Noncancerous tumor.<br>"
            "Malignant: Cancerous tumor.<br>"
            "Normal: No tumor detected.",
            self
        )
        explanation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        explanation_label.setStyleSheet("font-size: 12px; color: #333333; margin-top: 20px;")
        layout.addWidget(explanation_label)
        
        self.setLayout(layout)

class AnotherWindow(QWidget):
    def __init__(self, model1, model2, model3):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.setWindowTitle("Lung Cancer Detector")
        self.setFixedSize(QSize(800, 700))
        self.setStyleSheet('background-color: #CCE5FF;')
        
        self.Input_area = QWidget(self)
        self.Input_area.setGeometry(300, 100, 350, 550)
        self.InputLayout = QVBoxLayout()

        self.Input_add_button = QPushButton("Add Image")
        self.Input_add_button.setFixedSize(150, 40)
        self.Input_add_button.setStyleSheet("background-color: #4682B4; color: white; font-size: 14px;")
        self.Input_add_button.clicked.connect(self.get_image_from_file)

        self.Input_predict_button = QPushButton("Predict")
        self.Input_predict_button.setFixedSize(150, 40)
        self.Input_predict_button.setStyleSheet("background-color: #5F9EA0; color: white; font-size: 14px;")
        self.Input_predict_button.clicked.connect(self.predict)

        self.InputLayout.addWidget(self.Input_add_button)
        self.InputLayout.addWidget(self.Input_predict_button)
        self.InputLayout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.Input_area.setLayout(self.InputLayout)
        
        self.filenames = []
        self.predicted = False

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
                        image_label = QLabel(self)
                        image_label.setPixmap(pixmap)
                        image_label.setScaledContents(True)
                        image_label.setGeometry(170, 80, 450, 450)
                        image_label.show()
                    else:
                        print("Error: Failed to load image:", filename)
        return

    def predict(self):
        self.predicted = True
        self.infor_outputs = []
        for filename in self.filenames:
            index, pred1, pred2, pred3 = self.make_predictions(filename)
            self.infor_outputs.append((index, pred1, pred2, pred3))
        
        for i in range(len(self.filenames)):
            index, pred1, pred2, pred3 = self.infor_outputs[i]
            messages = [
                f'<b>Xception Prediction:</b> {pred1}',
                f'<b>InceptionResNetV2 Prediction:</b> {pred2}',
                f'<b>MobileNetV2 Prediction:</b> {pred3}'
            ]
            if index == 0:
                messages.append('<b>Overall Prediction:</b> Benign')
            elif index == 1:
                messages.append('<b>Overall Prediction:</b> Malignant')
            elif index == 2:
                messages.append('<b>Overall Prediction:</b> Normal')
            else:
                messages.append('<b>Overall Prediction:</b> Invalid prediction')
            self.open_new_window(messages)

    def open_new_window(self, messages):
        self.new_window = ShowResultWindow(messages)
        self.new_window.show()

    def make_predictions(self, image_path):
        try:
            img = load_img(image_path, target_size=(256, 256))
            img_array = img_to_array(img)
            scaled_img_array = img_array / 255.0
        except FileNotFoundError:
            print(f"Error: File not found: {image_path}")
            return None

        predictions1 = self.model1.predict(np.array([scaled_img_array]), verbose=0)
        predictions2 = self.model2.predict(np.array([scaled_img_array]), verbose=0)
        predictions3 = self.model3.predict(np.array([scaled_img_array]), verbose=0)

        RC1 = 2 * (1 - np.power(2, predictions1[0] - 1))
        RC2 = 2 * (1 - np.power(2, predictions2[0] - 1))
        RC3 = 2 * (1 - np.power(2, predictions3[0] - 1))
        FRS = RC1 + RC2 + RC3
        CCFS = 1 - (1/3 * (predictions1[0] + predictions2[0] + predictions3[0]))
        FDS = FRS * CCFS

        min_index = np.argmin(FDS)

        max1_index = np.argmax(predictions1[0])
        max2_index = np.argmax(predictions2[0])
        max3_index = np.argmax(predictions3[0])

        pred1 = ["Benign", "Malignant", "Normal"][max1_index]
        pred2 = ["Benign", "Malignant", "Normal"][max2_index]
        pred3 = ["Benign", "Malignant", "Normal"][max3_index]

        return min_index, pred1, pred2, pred3

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Cancer Detector")
        self.setFixedSize(QSize(800, 700))

        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, 800, 700)
        self.background_label.setPixmap(QPixmap('lung.png'))
        self.background_label.setScaledContents(True)

        self.Main = QWidget(self)
        self.Main.setGeometry(300, 100, 350, 550)
        self.Main_Layout = QVBoxLayout()

        self.button = QPushButton("Start")
        self.button.setStyleSheet('background-color: #4682B4; color: white; font-size: 16px;')
        self.button.setCheckable(True)
        self.button.clicked.connect(self.show_new_window)
        self.button.setFixedSize(150, 40)
        self.Main_Layout.addWidget(self.button)
        self.Main_Layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
    
        self.Main.setLayout(self.Main_Layout)
        self.Main.show()

        self.model1 = load_model('models/imageclassifier1.keras')
        self.model2 = load_model('models/imageclassifier2.keras')
        self.model3 = load_model('models/imageclassifier3.keras')

    def show_new_window(self, checked):
        self.w = AnotherWindow(self.model1, self.model2, self.model3)
        self.w.show()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

