import cv2
import numpy as np
import tensorflow as tf
from tensorflow import nn
from keras import utils

def predict_image_class(model_path, class_names):
    # Carregar o modelo treinado
    model = tf.keras.models.load_model(model_path)
    
    # Obter o formato da entrada
    input_shape = model.layers[0].input_shape

    # Verificar se o formato da entrada é uma lista
    if len(input_shape) == 4:
        input_shape = input_shape[1:3]  # Ignorar a dimensão do lote (batch)

    # Iniciar a captura de vídeo da webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Ler um quadro de vídeo da webcam
        ret, frame = cap.read()

        # Redimensionar o quadro para o formato de entrada do modelo
        frame = cv2.resize(frame, input_shape)

        # Pré-processar o quadro de acordo com o modelo
        img_array = utils.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar a previsão
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)  # Obter o índice da classe com maior probabilidade
        score = nn.softmax(prediction[0])
        confidence = 100 * np.max(score)

        # Obter o nome da classe prevista
        predicted_class_name = class_names[predicted_class]

        # Exibir o quadro com o resultado da previsão
        tf_score = tf.print(score)
        cv2.putText(frame, f"{class_names[np.argmax(score)]}({confidence:.2f} %)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        # Aguardar pela tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos da webcam e fechar as janelas
    cap.release()
    cv2.destroyAllWindows()

    return (predicted_class_name, predicted_class), confidence


model_path = "Sequential_299.h5"
class_names = ["Banana", "Laranja", "Maca", "Manga", "Morango"]
predict_image_class(model_path, class_names)