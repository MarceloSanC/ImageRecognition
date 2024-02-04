import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

start_time = datetime.datetime.now()
########################################################################################
#### Preprocess data

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices('GPU'))

data_dir = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Dataset\\Train"
val_dir = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Dataset\\Val"

model_name = 'MobileNet_8-7'
 
initial_epochs = 7

SEED = 1
SIZE = 224
BATCH_SIZE = 16
IMG_SIZE = (SIZE, SIZE)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255,
                                                                shear_range=0.2,
                                                                zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(data_dir,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    target_size=(SIZE, SIZE))

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
validation_generator = validation_datagen.flow_from_directory(val_dir, 
                                                              shuffle=True, 
                                                              batch_size=BATCH_SIZE, 
                                                              class_mode='categorical', 
                                                              target_size=(SIZE, SIZE))

class_names = list(train_generator.class_indices.keys())
print('Classes:', class_names)

########################################################################################
#### Model Architecture

IMG_SHAPE = IMG_SIZE + (3,)

# MobileNet trained dataset: 1.4M images; 1000 classes
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

base_learning_rate = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

loss0, accuracy0 = model.evaluate(validation_generator)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

########################################################################################
#### Train the model and see results

history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=validation_generator)

model.save(model_name)

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
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# fine_tune_epochs = 5
# total_epochs =  initial_epochs + fine_tune_epochs

# history_fine = model.fit(train_generator,
#                          epochs=1,
#                          validation_data=validation_generator)

# ########## Save file name ##########

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

end_time = datetime.datetime.now()
formatted_time = str(end_time - start_time).split(".")[0]
print("Tempo de execução:", formatted_time, '\n\n')