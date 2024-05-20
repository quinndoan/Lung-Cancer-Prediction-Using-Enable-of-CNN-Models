import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import load_model      #Load model 2
import numpy as np

#@title Choosing Resources

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)
   
import os
import cv2
import imghdr

image_exts = ['jpg', 'jpeg', 'png', 'bmp']
image_path = 'test4.png'
try:
    tip = imghdr.what(image_path)  # xác định định dạng hình ảnh
    if tip not in image_exts:
        print('Image not in ext list {}'.format(image_path))
    else:
        img = cv2.imread(image_path)
        # Thực hiện xử lý trên ảnh ở đây
except Exception as e:
    print('Issue with image {}'.format(image_path))




# Load ảnh từ đường dẫn và chuyển đổi thành mảng numpy
img = load_img(image_path, target_size=(256, 256))  # Thay target_size bằng kích thước bạn muốn
img_array = img_to_array(img)


# Scale the image
scaled_img_array = img_array / 255.0

# Check the maximum value after scaling
print("Maximum value after scaling:", scaled_img_array.max())

# Load model 1 with keras

# Đường dẫn tới tệp mô hình đã lưu
model_path = 'D:/really/models/imageclassifier1.keras'

# Tải lại mô hình
model1 = load_model(model_path)


# Đường dẫn tới tệp mô hình đã lưu
model_path2 = 'D:/really/models/imageclassifier2.keras'

# Tải lại mô hình
model2 = load_model(model_path2)

model_path3 = 'D:/really/models/imageclassifier3.keras'

# Tải lại mô hình
model3 = load_model(model_path3)

# Make prediction
predictions1 = model1.predict(np.array([scaled_img_array]))

# Interpret prediction
for pred in predictions1:
    max_index = np.argmax(pred)  # Get the index of the highest value in pred
    if max_index == 0:
        print("The image is predicted to be Bengin.")
    elif max_index == 1:
        print("The image is predicted to be Malginant.")
    elif max_index == 2:
        print("The image is predicted to be normal.")
        
predictions2 = model2.predict(np.array([scaled_img_array]))

# Interpret prediction
for pred in predictions2:
    max_index = np.argmax(pred)  # Get the index of the highest value in pred
    if max_index == 0:
        print("The image is predicted to be Bengin.")
    elif max_index == 1:
        print("The image is predicted to be Malginant.")
    elif max_index == 2:
        print("The image is predicted to be normal.")

predictions3 = model3.predict(np.array([scaled_img_array]))

# Interpret prediction
for pred in predictions3:
    max_index = np.argmax(pred)  # Get the index of the highest value in pred
    if max_index == 0:
        print("The image is predicted to be Bengin.")
    elif max_index == 1:
        print("The image is predicted to be Malginant.")
    elif max_index == 2:
        print("The image is predicted to be normal.")
print(predictions1[0])
print(predictions2[0])
print(predictions3[0])


# Định nghĩa một mảng NumPy toàn 0 với shape (kích thước) (n,)
n = 5
RC1 = np.zeros(3)
RC2 = np.zeros(3)
RC3 = np.zeros(3)

for i in range(len(RC1)):
    RC1[i] = 2 *(1 - pow(2,predictions1[0][i]-1))
    RC2[i] = 2*(1 - pow(2,predictions2[0][i]-1))
    RC3[i] = 2*(1 - pow(2,predictions3[0][i]-1))
    
print(RC1)
print(RC2)
print(RC3)

FRS = np.zeros(3)
for i in range(len(FRS)):
    FRS[i] = RC1[i] + RC2[i] + RC3[i]

CCFS = np.zeros(3)

CCFS[0] = 1 - (1/3 * (predictions1[0][0] + predictions2[0][0] + predictions3[0][0]))
CCFS[1] = 1 - (1/3 * (predictions1[0][1] + predictions2[0][1] + predictions3[0][1]))
CCFS[2] = 1 - (1/3 * (predictions1[0][2] + predictions2[0][2] + predictions3[0][2]))

FDS = np.zeros(3)
for i in range(len(FDS)):
    FDS[i] = FRS[i] * CCFS[i]

min_index = 100000000
index = 5
for i in range(len(FDS)):
      # Get the index of the highest value in pred
      if min_index > FDS[i]:
          min_index = FDS[i]
          index = i
          
          
if index == 0:
    print("The image is predicted to be Bengin.")
elif index == 1:
    print("The image is predicted to be Malginant.")
elif index == 2:
    print("The image is predicted to be normal.")    
