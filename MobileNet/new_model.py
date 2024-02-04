import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import datetime

start_time = datetime.datetime.now()
########################################################################################
#### Preprocess data

data_dir = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Dataset\\Train"
val_dir = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Dataset\\Val"

SEED = 1
SIZE = 224
BATCH_SIZE = 8
IMG_SIZE = (SIZE, SIZE)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    seed=SEED,
    batch_size=BATCH_SIZE,
    image_size=(SIZE, SIZE),
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    seed=SEED,
    batch_size=BATCH_SIZE,
    image_size=(SIZE,SIZE),
    shuffle=True,
)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

class_names = train_ds.class_names
print('Classes:', class_names)

# Ref: https://www.tensorflow.org/guide/data_performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

########################################################################################
#### Model Architecture

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names), activation='softmax')

IMG_SHAPE = IMG_SIZE + (3,)

# MobileNet trained dataset: 1.4M images; 1000 classes
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

inputs = tf.keras.Input(shape=(SIZE, SIZE, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

base_model.summary()
model.summary()

initial_epochs = 15

loss0, accuracy0 = model.evaluate(val_ds)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

########################################################################################
#### Train the model and see results

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


########################################################################################
#### Fine-tuning 

# base_model.trainable = True

# fine_tune_at = 144
# for layer in base_model.layers[:fine_tune_at]:
#   layer.trainable = False
  
# model.summary()

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['accuracy'])

# fine_tune_epochs = 5
# total_epochs =  initial_epochs + fine_tune_epochs

# history_fine = model.fit(train_ds,
#                          epochs=total_epochs,
#                          validation_data=val_ds)

# acc += history_fine.history['accuracy']
# val_acc += history_fine.history['val_accuracy']

# loss += history_fine.history['loss']
# val_loss += history_fine.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0.8, 1])
# plt.plot([initial_epochs-1,initial_epochs-1],
#           plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1.0])
# plt.plot([initial_epochs-1,initial_epochs-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

# loss, accuracy = model.evaluate(test_ds)
# print('Test accuracy :', accuracy)

########## Save file name ##########
model.save('MobileNet_8/')

end_time = datetime.datetime.now()
formatted_time = str(end_time - start_time).split(".")[0]
print("Tempo de execução:", formatted_time, '\n\n')