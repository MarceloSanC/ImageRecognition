import os, sys, pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mytools.dataset import treat_dataset
from mytools.lr_functions import logistic_regression_model, plot_learning_curve, test_image

true_path = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\True'
false_path = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\False'

X_train, Y_train, X_test, Y_test = treat_dataset(true_path , false_path, 64, 64, print_shapes=True)
trained_model = logistic_regression_model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.003, print_steps=True)

# print('trained_model: ', trained_model)

with open('logistic_regression_parameters.pkl', 'wb') as f:
   pickle.dump(trained_model ,f)

plot_learning_curve(trained_model)