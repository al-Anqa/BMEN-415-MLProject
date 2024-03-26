import os
import numpy as np
import cv2

def preprocess_images(folder, label, image_size=(224, 224)):
    data = []
    labels = []

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


print(x, y)     

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
