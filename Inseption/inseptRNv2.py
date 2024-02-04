import matplotlib.pyplot as plt
import numpy as np
import PIL, datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout

start_time = datetime.datetime.now()

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPSs Availabler: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = (299, 299)

model_save_path = 'InspetionRNv2_10'

train_ds = tf.keras.utils.image_dataset_from_directory(
    'C:\\Users\\marce\\Documents\\Code\\Datasets\\Train',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    seed=1,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'C:\\Users\\marce\\Documents\\Code\\Datasets\\Dev',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    seed=1,
)

class_names = train_ds.class_names
print('Class names:', class_names)

# Visualize data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

img_height, img_width = IMAGE_SIZE

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

base_model = keras.applications.InceptionResNetV2(
    weights='imagenet',
    input_shape=(img_height, img_width, 3),
    include_top=False,
)

base_model.treinable = False
inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)

# norm_layer = keras.layers.experimental.preprocessing.Normalization()
# mean = np.array([255.] * 3)
# var = mean ** 2

# x = norm_layer(x)
# norm_layer.set_weights([mean, var])

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(len(class_names))(x)
model = keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

base_model.trainable = True
model.summary()

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS_UNFREEZE = 5

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS_UNFREEZE
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS_UNFREEZE)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save(model_save_path)

end_time = datetime.datetime.now()
formatted_time = str(end_time - start_time).split(".")[0]
print("Tempo de execução:", formatted_time, '\n\n')