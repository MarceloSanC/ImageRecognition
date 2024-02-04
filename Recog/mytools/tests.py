import os
import sys
import PIL
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import read_images_rgb, treat_dataset
from lr_functions import test_image_lr, test_learning_rates_LR
from dl_functions import test_image_dl

#Use the parameters of the trained model in logisticregression.py to test an image
# with open('logistic_regression_parameters.pkl', 'rb') as f:
#    trained_model = pickle.load(f)

# my_image = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\test\\0142.jpg' 
# test_image_lr(my_image, trained_model)



# #Train the model with diferent learning rates
# true_path = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\True'
# false_path = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\False'

# learning_rates = [0.005, 0.006, 0.007, 0.0055]

# X_train, Y_train, X_test, Y_test = treat_dataset(true_path , false_path, 64, 64, print_shapes=True)
# test_learning_rates_LR(learning_rates, X_train, Y_train, X_test, Y_test, num_iterations=3000)

#Deep Learning Tests

print("Versão do TensorFlow:", tf.__version__)
print("Versão do Keras:", tf.keras.__version__)
print("Versão do NumPy:", np.__version__)