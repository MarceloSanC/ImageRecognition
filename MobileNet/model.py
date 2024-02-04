import numpy as np
import cv2
import PIL.Image as Image
import os
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import pathlib

from tensorflow import keras
from keras import layers
from keras.models import Sequential

from sklearn.model_selection import train_test_split

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

data_dir = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Dataset\\Train'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print('Imagens', image_count)

fruits_images_dict = {
    'banana': list(data_dir.glob('banana/*')),
    'laranja': list(data_dir.glob('laranja/*')),
    'maca': list(data_dir.glob('maca/*')),
    # 'manga': list(data_dir.glob('manga/*')),
    # 'morango': list(data_dir.glob('morango/*')),
}

fruits_labels_dict = {
    'banana': 0,
    'laranja': 1,
    'maca': 2,
    # 'manga': 3,
    # 'morango': 4,
}

# print(str(fruits_images_dict['banana'][0]))

print('Classes: ',len(fruits_images_dict))

X, y = [], []

for fruit_name, images in fruits_images_dict.items():
    for image in images:
        # print(image)
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (224,224))
        X.append(resized_img)
        y.append(fruits_labels_dict[fruit_name])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

num_of_fruits = len(fruits_images_dict)

model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    tf.keras.layers.Dense(num_of_fruits)
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)

model.fit(X_train_scaled, y_train, epochs=20)

model.save('MobileNet_3')