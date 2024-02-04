import os
from PIL import Image
from pathlib import Path
import imghdr

def convert_images_to_jpeg(directory):
    # Percorrer os arquivos e subdiretórios no diretório
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Verificar se o arquivo é uma imagem JPG
            if file.lower().endswith('.jpg', '.png'):
                # Caminho completo para o arquivo
                image_path = os.path.join(root, file)

                # Abrir a imagem usando o Pillow
                image = Image.open(image_path)

                # Gerar o novo nome de arquivo com a extensão .jpeg
                new_image_path = os.path.splitext(image_path)[0] + '.jpeg'

                # Converter a imagem para o formato JPEG e salvar
                image.convert('RGB').save(new_image_path, 'JPEG')

                # Excluir o arquivo de imagem JPG original
                os.remove(image_path)

def listar_tipos_arquivos(diretorio):
    tipos_arquivos = set()
    
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            _, ext = os.path.splitext(file)
            tipos_arquivos.add(ext)
    
    return tipos_arquivos

def encontrar_arquivos_invalidos(diretorio):
    arquivos_invalidos = []
    
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            nome_arquivo, ext = os.path.splitext(file)
            if ext.lower() not in ['.png', '.jpeg']:
                caminho_arquivo = os.path.join(root, file)
                arquivos_invalidos.append(caminho_arquivo)
    
    return arquivos_invalidos

def encontrar_arquivos_corrompidos(directory):
    corrupted_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                corrupted_files.append(file_path)
    return corrupted_files

def verificar_e_remover_imagens_corrompidas(diretorio):
    imagens_corrompidas = []
    
    for root, _, files in os.walk(diretorio):
        for file in files:
            arquivo = os.path.join(root, file)
            
            try:
                # Abre a imagem
                imagem = Image.open(arquivo)
                
                # Verifica se a imagem está vazia
                if imagem.size[0] == 0 or imagem.size[1] == 0:
                    imagens_corrompidas.append(arquivo)
                
                # Verifica se a imagem tem dimensões inválidas (por exemplo, muito pequena)
                # Substitua as dimensões mínimas e máximas pelo valor apropriado para o seu modelo
                if imagem.size[0] < 32 or imagem.size[1] < 32 or imagem.size[0] > 512 or imagem.size[1] > 512:
                    imagens_corrompidas.append(arquivo)
                
            except (IOError, SyntaxError) as e:
                # Se ocorrer um erro ao abrir a imagem, considera-se como corrompida
                imagens_corrompidas.append(arquivo)
                print(f"Erro ao abrir a imagem: {arquivo}")
    
    if len(imagens_corrompidas) > 0:
        print("Imagens corrompidas encontradas:")
        for imagem in imagens_corrompidas:
            print(imagem)
    else:
        print("Não foram encontradas imagens corrompidas.")
    print('Imagens corrompidas:', len(imagens_corrompidas))

    # Remove as imagens corrompidas
    for caminho in imagens_corrompidas:
        try:
            os.remove(caminho)
            print(f"Imagem removida: {caminho}")
        except OSError as e:
            print(f"Erro ao remover imagem: {caminho}, {e}")
    
    return

def culpir_files(data_dir):
    image_extensions = [".png", ".jpg", ".jpeg"]  # add there all your images file extensions

    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
    return


diretorio = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Test"
culpir_files(diretorio)

# verificar_e_remover_imagens_corrompidas(diretorio)

# arquivos_invalidos = encontrar_arquivos_invalidos(diretorio)
#########################
# print('Arquivos inválidos encontrados:')
# for arquivo in arquivos_invalidos:
#     print(arquivo)
#########################
# diretorio = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Dev"
# tipos = listar_tipos_arquivos(diretorio)

# print('Tipos de arquivos encontrados:')
# for tipo in tipos:
#     print(tipo)
#########################
# convert_images_to_jpeg("C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Train")
# convert_images_to_jpeg("C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Dev")
# convert_images_to_jpeg("C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Test")
#########################
# directory = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Test"
# corrupted_files = encontrar_arquivos_corrompidos(directory)
# print('Arquivos corrompidos encontrados:')
# for file in corrupted_files:
#     print(file)
#########################
