import tensorflow as tf
import numpy as np
from tensorflow import nn
from keras import utils


def predict_image_class(image_path, model_path, class_names):
    # Carregar o modelo treinado
    model = tf.keras.models.load_model(model_path)
    
    # Obter o formato da entrada
    input_shape = model.layers[0].input_shape
    # Verificar se o formato da entrada é uma lista
    if len(input_shape) == 4:
        input_shape = input_shape[1:3]  # Ignorar a dimensão do lote (batch)

    # Carregar e pré-processar a imagem de teste
    img = utils.load_img(image_path, target_size=input_shape)
    img_array = utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = img_array / 255.0  # Normalizar os valores de pixel

    # Realizar a previsão
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Obter o índice da classe com maior probabilidade
    score = nn.softmax(prediction[0])
    confidence = 100 * np.max(score)

    # Obter o nome das classes a partir do modelo
    # class_names = model.layers[-1].get_weights()[1]  # Última camada Dense do modelo
    predicted_class_name = class_names[predicted_class]

    # Imprimir os resultados
    tf.print(score)
    print(f"\nClass predicted as {class_names[np.argmax(score)]} with a {confidence:.2f} % confidence.\n")

    return (predicted_class_name, predicted_class), confidence

image_path = "C:\\Users\\marce\\Documents\\Code\\Datasets\\Test\\f2.png"
model_path = "Sequential_128.h5"
class_names = ["Banana", "Laranja", "Maca", "Manga", "Morango"]
predict_image_class(image_path, model_path, class_names)