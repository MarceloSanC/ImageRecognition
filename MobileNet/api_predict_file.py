import requests
import numpy as np
from PIL import Image
import random
import json

def enviar_imagem(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image_array = np.array(image)
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
            return result
        else:
            print("Erro ao enviar a imagem para a API. Status code:", response.status_code)
            return None
    except Exception as e:
        print("Erro ao enviar a imagem para a API:", str(e))
        return None

image_path = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Dataset\\Test\\800.jpg"

result = enviar_imagem(image_path)
if result:
    print(result)
else:
    print("Não foi possível obter uma resposta da API.")


















































 
def send_list():
    values = [[random.uniform(5, 50), random.uniform(5, 50), random.uniform(5, 50)] for _ in range(3)]
    print(type(values))

    response = requests.post("http://127.0.0.1:8000/receive_list/", json=values)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print("Erro ao enviar a lista para o servidor. Status code:", response.status_code)
        return None

# result = send_list()
# if result:
#     print("Tamanho da lista recebida:", result["tamanho"])
