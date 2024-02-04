import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mytools.dataset import treat_dataset, runtime
from mytools.dl_functions import deep_learning_model

start = time.time()

true_path = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\True'
false_path = 'C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\False'

layers_dims = [500, 100, 5, 1]

data = treat_dataset(true_path , false_path, print_shapes=True)
model = deep_learning_model(data, layers_dims, print_steps=True)

end = time.time()
runtime(start, end)