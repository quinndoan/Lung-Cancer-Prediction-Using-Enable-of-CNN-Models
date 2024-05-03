import sys
import re
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QPushButton, QDialogButtonBox, QLabel, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit, QFileDialog
from PyQt6.QtGui import QPalette, QColor, QIcon, QPixmap, QPainter, QPolygonF
from PyQt6.QtCore import QSize, Qt, QPointF
import pandas as pd
from pathlib import Path

def alert(message:str, parent=None):
	dlg = QMessageBox(parent)
	dlg.setWindowTitle("Message from the Application")
	dlg.setText(message)
	dlg.exec()

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
        self.Input_predict_button.clicked.connect(self.get_image_from_file)

        self.InputLayout.addWidget(self.Input_add_button)
        self.InputLayout.addWidget(self.Input_predict_button)
        self.InputLayout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.Input_area.setLayout(self.InputLayout)
        self.Input_area.show()
    
        #----------------------------App End------------------------------

    def get_image_from_file(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(str(Path('./')))
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                for filename in filenames:
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

window = MainWindow()
window.show()

app.exec()