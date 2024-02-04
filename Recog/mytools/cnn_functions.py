import os, sys, cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from keras.models import load_model
from PIL import Image, ImageOps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mytools')))

from mytools.cnn_models import convolutional_block, ResNet50

SIZE = 128
EPOCHS = 20

def cnn_model(train_directory_path, test_directory_path):
    train_directory_path = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Train"
    X_train, Y_train = process_images(train_directory_path)
    print('X_train:', X_train.shape, ',Y_train:', Y_train.shape)

    
    X_test, Y_test = process_images(test_directory_path)
    print('X_test:', X_test.shape, ',Y_test:', Y_test.shape)

    input_shape = (SIZE, SIZE, 3)
    # Models
    # model = convolutional_model(input_shape, Y_test.shape[1])
    model = ResNet50(input_shape, Y_test.shape[1])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset)
    model.save('H5s\\resnet50_'+str(EPOCHS)+'ep_'+str(SIZE)+'sz.h5')

    # with open('cnn.pkl', 'wb') as f:
    #         pickle.dump(model ,f)

    # df_loss_acc = pd.DataFrame(history.history)
    # df_loss= df_loss_acc[['loss','val_loss']]
    # df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    # df_acc= df_loss_acc[['accuracy','val_accuracy']]
    # df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    # df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    # df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    
    return model

def process_images(directory_path):
    fruit_classes = sorted(os.listdir(directory_path))
    num_classes = len(fruit_classes)

    images = []
    labels = []

    for class_index, fruit_class in enumerate(fruit_classes):
        class_dir = os.path.join(directory_path, fruit_class)
        image_files = os.listdir(class_dir)

        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            image = convert_and_resize(image_path)
            images.append(image)
            label = np.zeros(num_classes)
            label[class_index] = 1
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    images, labels = shuffle(images, labels)

    return images, labels

def convert_and_resize(image_path, size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

def frame2numpy(frame, size):
    #Convert frame to a PIL image
    pil_image = Image.fromarray(frame)
    image_data = ImageOps.exif_transpose(pil_image)
    image_data = image_data.convert("RGB")
    image_data = image_data.resize((size, size))
    
    # Convert PNG image to numpy array
    image = np.array(image_data) / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def predict_image_category(image, model):
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    confidence = 100 * np.max(score)
    
    return predictions, score, confidence

def process_test_image(image_path, model_path, size=128):
    image = convert_and_resize(image_path, size)
    image = np.expand_dims(image, axis=0)

    model = load_model(model_path, compile=False)
    predictions, score, confidence = predict_image_category(image, model)
    print(f"Image class predicted as {np.argmax(score)} with a {confidence:.2f} percent confidence.")
    return

def process_webcam_image(model_path, size=128):
    # Disable scientific notation
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model_path, compile=False)

    # Load the labels
    class_names = open("Recog\\mytools\\labels.txt", "r").readlines()
    cap = cv2.VideoCapture(0)

    while True:
        #Capture frames
        ret, frame = cap.read()

        image = frame2numpy(frame, size)
        predictions, score, confidence = predict_image_category(image, model)
        class_name = class_names[np.argmax(score)]
        
        cv2.putText(frame, f"Predictes as class ({class_name}) ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return
