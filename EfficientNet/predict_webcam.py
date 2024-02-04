import cv2
import tensorflow as tf
import keras
import time
import pyodbc
from keras.preprocessing import image
from keras.applications import imagenet_utils
import numpy as np

def request_price(cursor, product):
    command = f"""SELECT Preco from Produtos
    WHERE Nome = '{str(product)}'"""
    cursor.execute(command)
    price = cursor.fatchone()[0]
    return price

def predict_image_class(model_path, class_names):
    prev_class = 'nenhum'
    start_time = time.time()
    item_list = []

    connection_data = ("Driver={SQL Server};"
                     "Server=DESKTOP-8RM56RP\SQLEXPRESS;"
                     "Database=Precos;")
    con = pyodbc.connect(connection_data)
    cursor = con.cursor()

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
        if predicted_class != 'nenhum':
            if prev_class == 'nenhum':
                start_time = time.time()
                just_registered_item = 'nenhum'
            elif prev_class == predicted_class and (time.time() - start_time) >= 2 and just_registered_item != predicted_class:
                    item_list.append(predicted_class)
                    print('Lista: ', item_list)
                    just_registered_item = predicted_class
                    start_time = time.time()

        prev_class = predicted_class
            
        tf.print(score)
        # tf.print(prediction)
        if confidence > 80:
            cv2.putText(frame, f"{predicted_class} ({confidence})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else :
             cv2.putText(frame, f"Identificando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        # Fechar lista de compras e iniciar uma nova após tecla 'n'
        if cv2.waitKey(1) & 0xFF == ord('n'):
            print('\n\nLISTA DE COMPRAS')
            print('============================================')
            for item in item_list:
                print(item, '\t\t | peso:', np.random.uniform(200, 1000),'g\t| R$:',"{:.2f}".format(request_price(cursor, item)))
            item_list.clear()

        # Aguardar pela tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    # Liberar os recursos
    cursor.close()
    con.close()
    cap.release()
    cv2.destroyAllWindows()

    return 

model_path = "EfficientNetV2S_8"
with open(model_path+'_classes', 'r') as file:
    class_names = [line.strip() for line in file]

predict_image_class(model_path, class_names)