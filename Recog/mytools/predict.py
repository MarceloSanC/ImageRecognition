import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mytools.cnn_functions import process_webcam_image, process_test_image

model_path="H5s\\resnet50_18.h5"
size = 160

test_type = input('Input (0-Test image; 1-Webcam) : ')
if test_type == "0":
    test_image = "C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\Test\\0014.jpeg"
    process_test_image(test_image, model_path, size)
else:
    process_webcam_image(model_path, size)
