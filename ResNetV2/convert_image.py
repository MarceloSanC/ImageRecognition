import os
from PIL import Image

def convert_images_to_png(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)
            if file_ext.lower() != ".png":
                try:
                    # Open and convert the image to PNG format
                    image = Image.open(file_path)
                    image = image.convert("RGBA")
                    new_file_path = os.path.join(root, f"{file_name}.png")
                    # Save the converted image
                    image.save(new_file_path, "PNG")
                    # Check if the converted image is corrupted
                    try:
                        Image.open(new_file_path).verify()
                        print(f"Image '{file}' converted successfully.")
                        os.remove(file_path)  # Remove the old image file
                    except (IOError, SyntaxError):
                        print(f"\nConverted image '{file}' is corrupted!!!\n")
                        os.remove(new_file_path)  # Remove the corrupted image file
                except (IOError, SyntaxError):
                    print(f"Image '{file}' is corrupted.")

directory_path = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Train"
convert_images_to_png(directory_path)