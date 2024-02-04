from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
import tempfile

SIZE = 128

# def predict_image_category(image_path, model):
#     image = convert_and_resize(image_path)
#     image = np.expand_dims(image, axis=0)  # Adiciona uma dimensão extra para corresponder ao formato de entrada do modelo

#     prediction = model.predict(image)
#     predicted_class_index = np.argmax(prediction)
#     confidence_score = prediction[0][index]
#     # print(f'\n\nImage predicted as ({predicted_class_index})')
    
#     return predicted_class_index, confidence_score

def predict_image_category(image, model):
    image_data = ImageOps.exif_transpose(pil_image)
    image_data = image_data.convert("RGB")
    image_data = image_data.resize((SIZE, SIZE))
    
    # Convert PNG image to numpy array
    image = np.array(image_data) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict image category
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    confidence_score = prediction[0][predicted_class_index]
    
    return predicted_class_index, confidence_score

def convert_and_resize(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("H5s\\resnet50_20ep_128sz.h5", compile=False)

# Load the labels
class_names = open("mytools\\labels.txt", "r").readlines()

#Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    #Capture frames
    ret, frame = cap.read()

    #Convert frame to a PIL image
    pil_image = Image.fromarray(frame)
    index, confidence = predict_image_category(pil_image, model)
    class_name = class_names[index]
    
    cv2.putText(frame, f"Predictes as class ({class_name}) ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break