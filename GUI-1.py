import sys
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QPushButton, QDialogButtonBox, QLabel, QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit, QFileDialog 
from PyQt6.QtGui import QPalette, QColor, QIcon, QPixmap, QPainter, QPolygonF 
from PyQt6.QtCore import QSize, Qt, QPointF  
import pandas as pd  
import numpy as np
from pathlib import Path
from keras.models import load_model

def make_predictions(image_path):
    # Preprocess image
    try:
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        scaled_img_array = img_array / 255.0
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
        return None  # Hoặc xử lý lỗi theo cách khác

    # Load models
    model_path1 = 'models/imageclassifier1.keras'
    model_path2 = 'models/imageclassifier2.keras'
    model_path3 = 'models/imageclassifier3.keras'
    try:
        model1 = load_model(model_path1)
        model2 = load_model(model_path2)
        model3 = load_model(model_path3)
    except FileNotFoundError:
        print("Error: Model file not found.")
        return None  # Hoặc xử lý lỗi theo cách khác

    # Make predictions (tối ưu hóa bằng vectorization)
    predictions1 = model1.predict(np.array([scaled_img_array]))
    predictions2 = model2.predict(np.array([scaled_img_array]))
    predictions3 = model3.predict(np.array([scaled_img_array]))

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

def alert(message:str, parent=None):#dc định nghĩa để hiện thị cửa sổ cho người dùng
	dlg = QMessageBox(parent)
	dlg.setWindowTitle("Message from the Application")
	dlg.setText(message)
	dlg.exec()

# class AboutDlg(QDialog):
# #Hiển thị thông tin bổ sung về ứng dụng
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Additional Information")
#         self.setFixedSize(1100,600)
#         manual=QLabel(self)
#         about_text=f'The program permits users to input CT scan images and forecast whether they exhibit signs of lung cancer'
#         manual.setText(about_text)
#         manual.show()
        
class AnotherWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lung cancer detector")
        self.setFixedSize(QSize(1418, 747))
        self.setStyleSheet('background-color: #CCE5FF;')

        #----------------------------App Start-----------------------------

        # Tạo layout ngang cho hai phần
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        # Phần bên trái: Hiển thị ảnh
        self.image_area = QWidget(self)
        self.image_layout = QVBoxLayout()
        self.image_area.setLayout(self.image_layout)
        self.main_layout.addWidget(self.image_area)

        # Phần bên phải: Hiển thị kết quả
        self.result_area = QWidget(self)
        self.result_layout = QVBoxLayout()
        self.result_area.setLayout(self.result_layout)
        self.main_layout.addWidget(self.result_area)

        # Tạo vùng chứa cho nút Add Image và nút Predict
        self.button_container = QWidget(self.result_area)
        self.button_layout = QHBoxLayout()
        self.button_container.setLayout(self.button_layout)
       
        # Thêm nút Add Image
        self.Input_add_button = QPushButton("Add image")
        self.Input_add_button.setCheckable(True)
        self.Input_add_button.setFixedSize(150, 40)
        self.Input_add_button.clicked.connect(self.get_image_from_file)
        self.button_layout.addWidget(self.Input_add_button)

        # Thêm nút Predict
        self.Input_predict_button = QPushButton("Predict")
        self.Input_predict_button.setCheckable(True)
        self.Input_predict_button.setFixedSize(150, 40)
        self.Input_predict_button.clicked.connect(self.predict)
        self.button_layout.addWidget(self.Input_predict_button)
        
         #Căn giữa button_container theo chiều ngang trong result_layout
        self.result_layout.setAlignment(self.button_container, Qt.AlignmentFlag.AlignHCenter)
        self.result_layout.addWidget(self.button_container)


        # Vùng hiển thị kết quả
        self.OutputList_scroll_area = QScrollArea(self.result_area)
        self.OutputList_scroll_area.setWidgetResizable(True)
        self.OutputList_area_layout = QVBoxLayout()
        self.OutputList_scroll_area.setLayout(self.OutputList_area_layout)
        # Đặt chiều cao tối thiểu cho khung hiển thị kết quả
        self.OutputList_scroll_area.setMinimumHeight(200) 
        self.result_layout.addWidget(self.OutputList_scroll_area)
        
        #----------------------------App End------------------------------

        self.filenames = []
        self.predicted = False

        # Đặt tỷ lệ co giãn cho hai phần bằng nhau
        self.main_layout.setStretch(0, 1)  # Phần bên trái
        self.main_layout.setStretch(1, 1)  # Phần bên phải
    def get_image_from_file(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(str(Path('./')))
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                self.filenames = filenames  # Lưu trữ danh sách file
                for filename in filenames:
                    pixmap = QPixmap(filename)
                    if not pixmap.isNull():
                        image_label = QLabel(self.image_area)
                        image_label.setPixmap(pixmap)
                        image_label.setScaledContents(True)
                        self.image_layout.addWidget(image_label)
                    else:
                        print("Error: Failed to load image:", filename)
        return

    # def predict(self):
    #     self.predicted = True
        
    #     self.infor_outputs = make_predictions(self.filenames)
    #     assert len(self.filenames) == len(self.infor_outputs)

    #     # Hiển thị kết quả dự đoán
    #     for i in range(len(self.filenames)):
    #         if self.infor_outputs[i] == 0:
    #             self.OutputList_area_layout.addWidget(QLabel(f'{self.filenames[i]}  -  Không có dấu hiệu ung thư phổi'))
    #         elif self.infor_outputs[i] == 1:
    #             self.OutputList_area_layout.addWidget(QLabel(f'{self.filenames[i]}  -  Có dấu hiệu nghi ngờ ung thư phổi'))
    #         else:
    #             self.OutputList_area_layout.addWidget(QLabel(f'{self.filenames[i]}  -  Có dấu hiệu ung thư phổi'))
    #     self.OutputList_scroll_area.update()

    def get_image_from_file(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(str(Path('./')))
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                self.filenames = filenames  # Lưu trữ danh sách file
                for filename in filenames:
                    pixmap = QPixmap(filename)
                    if not pixmap.isNull():
                        image_label = QLabel(self.image_area)
                        image_label.setPixmap(pixmap)
                        image_label.setScaledContents(True)
                        self.image_layout.addWidget(image_label)
                    else:
                        print("Error: Failed to load image:", filename)
        return

    def predict(self):
        self.predicted = True

        # Call make_predictions() to generate predictions
        self.infor_outputs = []  # Initialize an empty list
        for filename in self.filenames:
            index = make_predictions(filename)
            self.infor_outputs.append(index)

        assert len(self.filenames) == len(self.infor_outputs)

        # Now display the results
        for i in range(len(self.filenames)):
            index = self.infor_outputs[i]
            if index == 0:
                self.OutputList_area_layout.addWidget(QLabel(f' The image is predicted to be Bengin.'))
            elif index == 1:
                self.OutputList_area_layout.addWidget(QLabel(f' The image is predicted to be Malginant.'))
            elif index == 2:
                self.OutputList_area_layout.addWidget(QLabel(f' The image is predicted to be normal.'))
            else:
                self.OutputList_area_layout.addWidget(QLabel(f' Invalid prediction.'))
        self.OutputList_scroll_area.update()
class MainWindow(QMainWindow):
#Lớp này đại diện cho cửa sổ chính ba đầu của ứng dụng
#trong lớp này, mọt hình ảnh của phổi (là hình ảnh mẫu) được hiển thị
    def __init__(self):
        super(MainWindow,self).__init__()
    
        self.setWindowTitle("Lung cancer detector")
        self.setFixedSize(QSize(1418, 747))

        # creating label
        self.label = QLabel(self)
         
        # loading image
        self.pixmap = QPixmap('lung.png')
 
        # adding image to label
        self.label.setPixmap(self.pixmap)
 
        # Optional, resize label to image size
        self.label.resize(1440,761)

        self.Main = QWidget(self)
        self.Main.setGeometry(150, 10, 1000, 750)
        self.Main_Layout = QVBoxLayout()

        self.button = QPushButton("Start")
        #nút "Starr" để mở cửa sổ khác 'AnotherWindow' khi được nhân
        #khi người dùng nhấn vào nút "Start", một cửa sổ mới sẽ hiển thị, cho phép họ thêm hình ảnh và thực hiện dự đoán
        self.button.setCheckable(True)
        self.button.setGeometry(150,50,600,400)
        self.button.clicked.connect(self.show_new_window)
        self.button.setFixedSize(150,40)
        self.Main_Layout.addWidget(self.button)

        
        #tên app
        # Tạo một QWidget làm vùng hiển thị chính
    
        
        # Tạo QLabel và đặt các thuộc tính
        widget = QLabel("Lung cancer detector", self.Main)
        font = widget.font()
        font.setPointSize(60)
        font.setBold(True)  # Đặt chữ in đậm
        widget.setFont(font)
        
        # Đặt màu chữ và đổ bóng bằng cách sử dụng CSS
        widget.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                text-align: center;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);  /* Đổ bóng */
            }
        """)
        
        
        # Đặt tọa độ vị trí hiển thị
        widget.setGeometry(00, 150, 670, 100)  # Điều chỉnh tọa độ và kích thước theo ý bạn

        #Note
        # Tạo QLabel và đặt các thuộc tính
        widget1 = QLabel("The program permits users to input CT scan images and forecast whether they exhibit signs of lung cancer.", self.Main)
        font = widget1.font()
        font.setPointSize(15)
        font.setItalic(True)
        # Đặt màu chữ
        widget1.setStyleSheet("color: white;")
        widget1.setFont(font)
        # Đặt căn chỉnh cho văn bản
        widget1.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        
        # Đặt tọa độ vị trí hiển thị
        widget1.setGeometry(00, 670, 800, 50)  # Điều chỉnh tọa độ và kích thước theo ý bạn
        
        self.Main.setLayout(self.Main_Layout)
        self.Main.show()
    
    def show_new_window(self, checked):
        self.hide()
    #hàm này được gọi khi nút "Start" được nhấn trong cửa sổ chính
    #nó tạo một lớp 'AnotherWindow' đẻ hiển thị nó
        self.w = AnotherWindow()
        self.w.show()


app = QApplication(sys.argv)
#app.setStyleSheet(stylesheet)

window = MainWindow()
window.show()

app.exec()