import os
import numpy as np
import cv2
from keras.layers import Activation, Dense
from keras.models import Sequential 
from sklearn.model_selection import train_test_split


def preprocess_images(folder, label, image_size=(224, 224)):
    data = []
    labels = []

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize pixel values to the range [0, 1]

        data.append(img)
        labels.append(label)

    return np.array(data), np.array(labels)

dataset_pos = 'datasets/brain_mri_scan_images/positive'
dataset_neg = 'datasets/brain_mri_scan_images/negative'

positive_data, positive_labels = preprocess_images(dataset_pos, 1)
negative_data, negative_labels = preprocess_images(dataset_neg, 0)

x = np.concatenate((positive_data, negative_data))
y = np.concatenate((positive_labels, negative_labels))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# print(x_train, x_test)
# print(y_train, y_test)

print(f'Train: X={x_train.shape}, y={y_train.shape}')
print(f'Test: X={x_test.shape}, y={y_test.shape}')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

print(f'Train: X={x_train.shape}, y={y_train.shape}')
print(f'Test: X={x_test.shape}, y={y_test.shape}')

model = Sequential()
model.add(Dense(32, input_dim=50176))
# model.add(Conv2D())
model.add(Activation('relu'))
model.add(Dense(10, input_dim=32))
model.add(Activation('softmax'))

model.summary()