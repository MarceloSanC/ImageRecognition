import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image

def logistic_regression_model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_steps=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (width * height * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (width * height * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_steps -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # Apply the train and test entries vector in the model to be trained
    w, b = initialize_wb(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_steps)
    w = params["w"]
    b = params["b"]
    
    # Test the accuracy of trained model with the train and test data
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    if print_steps:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    results = {"costs": costs, 
               "Y_prediction_test": Y_prediction_test, 
               "Y_prediction_train" : Y_prediction_train, 
               "w" : w, 
               "b" : b, 
               "learning_rate" : learning_rate, 
               "num_iterations": num_iterations}
    
    return results

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))

def initialize_wb(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """
    
    w = np.random.randn(dim, 1) * np.sqrt(2/dim)
    b = 0.0
    return w, b

def optimize(w, b, X, Y, num_iterations, learning_rate, print_steps=True):
    """
    This function optimizes w and b by running a gradient descent algorithm
    Use trained w and b to predict the labels for a dataset 
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    if print_steps: print(f'w shape: {w.shape}')
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    
    # Train the model to optimize the best w and b values with n number of iterations
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            if print_steps: print("Cost after iteration %i: %f" %(i, cost))
            
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1]
    
    # Calcualte the linear transformation and the cost
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

    # Calculate the gradients of dJ/dw and dJ/db
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)
    
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def predict(w, b, X, min_confidence=0.5):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    min_confidence -- minimum of confidence to predict as true
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    # Initialize the prediciton matrix for m training entries
    m = X.shape[1] 
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Calculates the linear tranformation with the trained parameters (w, b)
    A = sigmoid(np.dot(w.T, X)+b)
    
    # Predict as true if a minumum confidence values has been achived 
    Y_prediction = (A > min_confidence).astype(int)
    
    return Y_prediction

def plot_learning_curve(trained_model):
    """Plot learning curve (with costs)

    Args:
        trained_model (dictionary): must contain the 'costs', 'learning_rate' of a trained model
    """
    costs = np.squeeze(trained_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(trained_model["learning_rate"]))
    plt.show()

def test_learning_rates_LR(learning_rates, train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=3000):
    """Compare the learning curve of a model with several choices of learning rates

    Args:
        learning_rates (list of floats): _description_
        train_set_x : training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        train_set_y : training labels represented by a numpy array (vector) of shape (1, m_train)
        test_set_x : test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        test_set_y : test labels represented by a numpy array (vector) of shape (1, m_test)
    """
    models = {}

    for lr in learning_rates:
        print ("Training a model with learning rate: " + str(lr))
        models[str(lr)] = logistic_regression_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=3000, learning_rate=lr, print_steps=False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for lr in learning_rates:
        plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    return

def test_image_lr(my_image, trained_model):
    """Preprocess the image to fit the algorithm.

    Args:
        my_image (path): path name of a image file
        trained_model (dictionary): must contain the parameters 'w' and 'b' of a trained model
    """
    
    image = np.array(Image.open(my_image).resize((64, 64)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, 64 * 64 * 3)).T
    my_predicted_image = predict(trained_model["w"], trained_model["b"], image)
    classes = np.array(['non-banana', 'banana'])

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + str(classes[int(np.squeeze(my_predicted_image)),]) +  "\" picture.")