import cv2
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications import imagenet_utils
import numpy as np


def predict_image_class(model_path, class_names):
    
    model = tf.keras.models.load_model(model_path)
    input_shape = model.layers[0].input_shape
    input_shape = input_shape[0][1:3]
    
    cap = cv2.VideoCapture(0)

    while True:
        # Ler um quadro de vídeo da webcam
        ret, frame = cap.read()

        # Redimensionar o quadro para o formato de entrada do modelo
        frame = cv2.resize(frame, input_shape)
        
        img_array = keras.utils.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 
        
        prediction = model.predict(img_array)
        # score = tf.nn.softmax(prediction[0])
        score = prediction[0]
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        # predicted_class_name = class_names[predicted_class]

        # Exibir o quadro com o resultado da previsão
        tf.print(score)
        tf.print(prediction)
        if confidence > 0:
            cv2.putText(frame, f"{predicted_class} ({confidence})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        # Aguardar pela tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()

    return 

model_path = "MobileNet_202/"
class_names = ["Banana", "Laranja", "Maca", "Manga", "Morango"]
predict_image_class(model_path, class_names)