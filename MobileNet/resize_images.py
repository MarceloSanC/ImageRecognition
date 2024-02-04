import os
import cv2

def resize_images(directory, target_size):
    # Verifica se o diretório existe
    if not os.path.isdir(directory):
        print(f'O diretório "{directory}" não existe.')
        return
    
    # Obtém a lista de subdiretórios no diretório principal
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
    
    # Itera sobre cada subdiretório
    for subdir in subdirectories:
        subdir_path = os.path.join(directory, subdir)
        
        # Obtém a lista de arquivos no subdiretório
        files = os.listdir(subdir_path)
        
        # Itera sobre cada arquivo
        for file in files:
            file_path = os.path.join(subdir_path, file)
            
            # Verifica se o arquivo é uma imagem
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Carrega a imagem utilizando o OpenCV
                image = cv2.imread(file_path)
                
                # Redimensiona a imagem para o tamanho desejado
                resized_image = cv2.resize(image, target_size)
                
                # Salva a imagem redimensionada no mesmo local com o mesmo nome
                cv2.imwrite(file_path, resized_image)
                
                print(f'A imagem "{file}" foi redimensionada para o tamanho {target_size}.')
    
# Exemplo de uso
directory_path = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Dataset\\Train"
target_size = (1008, 756)
resize_images(directory_path, target_size)
