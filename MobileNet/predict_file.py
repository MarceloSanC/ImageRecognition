import os, cv2
import tensorflow as tf
import numpy as np
from keras import utils

def predict_image_class(image_path, model_path, class_names):
    # Carregar o modelo treinado
    model = tf.keras.models.load_model(model_path)
    
    # Obter o formato da entrada
    input_shape = model.layers[0].input_shape
    input_shape = input_shape[1:3]
    # input_shape = input_shape[0][1:3]

    # Carregar e pré-processar a imagem de teste
    img = utils.load_img(image_path, target_size=input_shape)
    img_array = utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 

    # Realizar a previsão
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0]
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Obter o nome das classes a partir do modelo
    # class_names = model.layers[-1].get_weights()[1]  # Última camada Dense do modelo
    # predicted_class_name = class_names[predicted_class]

    # Imprimir os resultados
    # print(prediction)
    # tf.print(score)
    print(f"Class predicted as {class_names[np.argmax(score)]} with a {confidence:.2f} % confidence.\n")

    return

def predict_dir_files(dir_path, model_path):
    with open(model_path+'_classes', 'r') as file:
        class_names = [line.strip() for line in file]
    
    image_files = os.listdir(dir_path)
    
    for image in image_files:
        image_path = os.path.join(dir_path, image)
        print("\nImage :",image)
        predict_image_class(image_path, model_path, class_names)
    
model_path = "MobileNet_8-7"
dir_path = "C:\\Users\\marce\\OneDrive\\Imagens\\Test"

predict_dir_files(dir_path, model_path)

# predict_image_class(image_path, model_path, class_names)