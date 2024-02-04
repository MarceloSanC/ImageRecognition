import requests
import numpy as np
from PIL import Image
import cv2

def enviar_imagem():
    try:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_pil = Image.fromarray(frame_rgb)
            image_array = np.array(image_pil)
            print('shape:', image_array.shape)
            
            red_channel = image_array[:, :, 0].tolist()
            green_channel = image_array[:, :, 1].tolist()
            blue_channel = image_array[:, :, 2].tolist()
            
            data = {
                "red_channel": red_channel,
                "green_channel": green_channel,
                "blue_channel": blue_channel
            }

            response = requests.post("http://127.0.0.1:8000/predict/", json=data)

            if response.status_code == 200:
                result = response.json()
                print(result)
            else:
                print("Erro ao enviar a imagem para a API. Status code:", response.status_code)

            cv2.imshow("Webcam", frame)

            # 'q' para encerrar a transmissão
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Erro ao enviar a imagem para a API:", str(e))

enviar_imagem()