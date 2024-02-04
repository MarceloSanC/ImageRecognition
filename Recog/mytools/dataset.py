import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def treat_dataset(true_dataset_path, false_dataset_path, width=64, height=64, print_shapes=False):
    """_summary_

    Args:
        true_dataset_path_list (string): List of paths to all directories that contain true images
        false_dataset_path_list (string): List of paths to all directories that contain false images
        width (int): width of the RGB image scale
        height (int): height of the RGB image scale
        print_shapes (bool, optional): Print the shapes of matrix as it is treated. Defaults to False.

    Returns:
        X_train (numpy.ndarray): training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train (numpy.ndarray): training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test (numpy.ndarray): test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test (numpy.ndarray): test labels represented by a numpy array (vector) of shape (1, m_test)
    """

    # Matrix with all images converted to RGB of shape (num_examples, 64, 64, 3)
    true_dataset = read_images_rgb(true_dataset_path, width, height)
    false_dataset = read_images_rgb(false_dataset_path, width, height)
    print(f'\nConverting {true_dataset.shape[0]+false_dataset.shape[0]} images to RGB of size {width}x{height}x3')
    print('True dataset size:',true_dataset.shape[0])
    print('False dataset size:',false_dataset.shape[0])


    # Initialize label matrix
    labels = create_labels(true_dataset, false_dataset)
    dataset = np.concatenate((true_dataset, false_dataset), axis=0)
    if print_shapes: 
        print('Dataset size:', dataset.shape[0])
        print('Labels shape:', labels.shape)

    # Split the dataset and the labels into shuffled train and test matrix
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels.T, test_size=0.3, random_state=42)
    Y_train = Y_train.T
    Y_test = Y_test.T

    # Flatten each image matrix to a 1 dim vector of widht*height*3 size
    X_train_flat = X_train.reshape(X_train.shape[0], -1).T
    X_test_flat = X_test.reshape(X_test.shape[0], -1).T

    if print_shapes: 
        print('Flat train set shape:', X_train_flat.shape)
        print('Flat test set shape:', X_test_flat.shape)
        print('----------------------------------------\n')

    # Standardize the dataset for values range from 0 to 1
    X_train = X_train_flat / 255
    X_test = X_test_flat / 255
    
    data = (X_train, Y_train, X_test, Y_test)
    return data

def read_images_rgb(directory, width, height):
    """_summary_

    Args:
        directory (_type_): _description_
        width (_type_): _description_
        height (_type_): _description_

    Returns:
        _type_: _description_
    """
    image_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((width, height))
                    img_array = np.array(img)
                    image_list.append(img_array)
    return np.array(image_list)

def create_labels(true_dataset, false_dataset):
    """Creates a (1, n+m) matrix of 1 and 0 based on the true or false dataset origin

    Args:
        true_dataset (numpy.ndarray): Matrix of shape (n, widht, height, 3) with all the true examples
        false_dataset (numpy.ndarray): Matrix of shape (m, widht, height, 3) with all the false examples

    Returns:
        labels: Concatenated label matrix with [1,1,...,0,0] corresponding for all given examples
    """
    ones = np.ones((1, true_dataset.shape[0]))
    zeros = np.zeros((1, false_dataset.shape[0]))
    labels = np.concatenate((ones, zeros), axis=1)
    return labels

def complete_path(path_list):
    """Complete the directory path to a (C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\) folder

    Args:
        path_list (string): The same of the directory to be accessed

    Returns:
        complete_path_list (string): The complete path from the root directory to a specific dataset directory
    """
    complete_path_list = []
    for path in path_list:
        complete_path_list.append('C:\\Users\\marce\\OneDrive\\Documentos\\Code\\Datasets\\'+str(path))

    return complete_path_list

def runtime(start, end):
    runtime = end - start
    hours, rest = divmod(runtime, 3600)
    mins, segs = divmod(rest, 60)

    # imprimir o tempo de execução
    print("Run time: {:02}:{:02}:{:02}".format(int(hours), int(mins), int(segs)))
    return