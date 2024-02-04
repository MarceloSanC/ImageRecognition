from PIL import Image

image_path = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Test\\6.png" # Replace with the actual image file path
image = Image.open(image_path)

if image is not None:
    # Image opened successfully
    # Perform further operations with the image here
    image.show()