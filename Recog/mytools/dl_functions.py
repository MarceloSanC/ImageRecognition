import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import floor

def deep_learning_model(data, layers_dims, learning_rate=0.007, num_iterations=500, mb_size=64, beta1=0.9, beta2=0.999, lambd=0.7, keep_prob=1, decay_rate=1, print_steps=False):
    """Train a Neural Network model with an architeture of L layers

    Args:
        data (list): X_train - Train set converted to RGB and shape of ( width * height * 3, num_samples ),
                     Y_train - Train set label of shape (1 , num_train_samples),
                     X_test - Test set converted to RGB and shape of ( width * height * 3, num_samples ),
                     Y_test - Train set label of shape (1 , num_test_samples)
        layers_dims (array): Array with the dimensions of the L layers
        learning_rate (float, optional): Learning rate of gradient descent. Defaults to 0.0075.
        num_iterations (int, optional): Num of epochs. Defaults to 3000.
        lambd (float, optional): Regularization parameter. Defaults to 0.7.
        keep_prob (float, optional): Dropout parameter. Defaults to 0.8.
        print_steps (bool, optional): Print steps of precessing. Defaults to False.

    Returns:
        tuple: 
            parameters (dictionary): Contain the trained parameters
            cost (array): Contain the cost of every 100 epochs
    """
    (X_train, Y_train, X_test, Y_test) = data
    layers_dims.insert(0, X_train.shape[0])
    parameters = initialize_wb_dn(layers_dims)
    costs = []

    print('Training Deep Learning Model:')
    print(f'{len(layers_dims)} layers of shape: {layers_dims}')
    print('learning rate = ',learning_rate)
    print('num_iterations = ',num_iterations)
    print('lambda = ',lambd)
    print('keep_prob =', keep_prob)
    print('----------------------------------------\n')

    parameters, costs = adam_model(X_train, Y_train, layers_dims, learning_rate, num_iterations, mb_size, beta1, beta2, lambd, keep_prob, decay_rate, print_steps)

    # for i in range(0, num_iterations):
    #     print(f'X_train.shape: {X_train.shape}')
    #     AL, caches, D_list = model_forward(X_train, parameters, keep_prob)
    #     cost = compute_cost(AL, Y_train, parameters, lambd)
    #     grads = model_backward(AL, Y_train, caches, D_list, lambd, keep_prob)
    #     parameters = update_parameters(parameters, grads, learning_rate)

    #     if print_steps and i % 100 == 0 or i == num_iterations - 1:
    #         print(f"Cost after iteration {i}: {np.squeeze(cost)}")
    #     if i % 100 == 0 or i == num_iterations - 1:
    #         costs.append(cost)

    pred_train = predict(X_train, Y_train, parameters, "Train", min_confidence=0.5)
    pred_test = predict(X_test, Y_test, parameters, "Test", min_confidence=0.5)

    
    plot_learning_curve(costs, learning_rate, layers_dims)

    with open('deep_learning.pkl', 'wb') as f:
        pickle.dump(parameters ,f)

    model = (parameters, costs)
    return model

def adam_model(X, Y, layers_dims, learning_rate, num_iterations, mb_size, beta1, beta2, lambd, keep_prob, decay_rate, print_steps):
    m = X.shape[1]
    costs = []
    t = 0
    seed = 0
    lr_rates = []
    learning_rate0 = learning_rate

    parameters = initialize_wb_dn(layers_dims)
    v, s = initialize_adam(parameters)

    for i in range(num_iterations):
        seed = seed + i
        mini_batches = random_mini_batches(X, Y, mb_size, seed)
        cost_total = 0

        for mini_batch in mini_batches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = mini_batch
            # Forward propagation
            AL, caches, D_list = model_forward(minibatch_X, parameters, keep_prob)
            # Compute mini batch cost and add to the total
            cost_total += compute_cost(AL, minibatch_Y, parameters, lambd)
            # Backward propagation
            grads = model_backward(AL, minibatch_Y, caches, D_list, lambd, keep_prob)
            # Update parameters
            t += 1 # Adam counter
            parameters, v, s, _, _, = update_parameters_with_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate)

        cost_avg = cost_total / m
        if decay_rate > 0:
            learning_rate = learning_rate0 /(1 + decay_rate * i)
        if print_steps and i % 100 == 0 or i == num_iterations - 1:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
            if decay_rate > 0:
                print("learning rate after epoch %i: %f"%(i, learning_rate))
        if i % 100 == 0 or i == num_iterations - 1:
            costs.append(cost_avg)

    return parameters, costs

def initialize_wb_dn(layers_dims):
    np.random.seed(0)
    parameters = {}

    for l in range(1, len(layers_dims)):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def model_forward(A, parameters, keep_prob):
    L = len(parameters)//2
    caches = []
    D_list = []

    for l in range(1, L):
        A_prev = A
        A, D, cache = linear_forward_activation(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu", keep_prob)
        D_list.append(D)
        caches.append(cache)

    AL, D, cache = linear_forward_activation(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid", keep_prob=1)
    caches.append(cache)
    D_list.append(D)
    
    return AL, caches, D_list

def linear_forward_activation(A_prev, W, b, activation, keep_prob):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)

    if activation == "sigmoid":
        A = np.array(sigmoid(Z))
    elif activation == "relu":
        A = np.array(relu(Z))

    # Dropout
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    A *= D/keep_prob
    
    cache = (linear_cache, Z)
    return A, D, cache

def compute_cost(A, Y, parameters, lambd):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(A),Y) + np.multiply(-np.log(1 - A), 1 - Y)
    normal_cost = 1./m * np.nansum(logprobs)

    L2_cost = np.sum(np.zeros((parameters["W1"].shape)))
    for l in range(1, len(parameters)//2):
        L2_cost += np.sum(np.square(parameters["W"+str(l)]))

    L2_cost *= lambd / (2*m)

    cost = normal_cost + L2_cost
    return cost

def model_backward(AL, Y, caches, D_list, lambd, keep_prob):
    grads = {}
    m = AL.shape[1]
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid", D_list[L-1], lambd, keep_prob=1)
    grads["dA"+str(L-1)] = dA_prev_temp
    grads["dW"+str(L)] = dW_temp
    grads["db"+str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu", D_list[l], lambd, keep_prob)
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp

    return grads

def linear_activation_backward(dA, cache, activation, D, lambd, keep_prob):

    linear_cache, activation_cache = cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dA_prev = dA * D / keep_prob

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dW = (1/m) * np.dot(dZ, A_prev.T) + (lambd/m)*W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dW = (1/m) * np.dot(dZ, A_prev.T) + (lambd/m)*W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def random_mini_batches(X, Y, mb_size, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    num_complete_mb = floor(m/mb_size)
    for k in range(0, num_complete_mb):
        mini_batch_X = shuffled_X[:, k*mb_size : (k+1)*mb_size]
        mini_batch_Y = shuffled_Y[:, k*mb_size : (k+1)*mb_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mb_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_mb*mb_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_mb*mb_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def initialize_adam(parameters):
    v = {}
    s = {}

    for l in range(1, (len(parameters)//2)+1):
        v["dW"+str(l)] = np.zeros((parameters['W'+str(l)].shape))
        v["db"+str(l)] = np.zeros((parameters['b'+str(l)].shape))
        s["dW"+str(l)] = np.zeros((parameters["W"+str(l)].shape))
        s["db"+str(l)] = np.zeros((parameters["b"+str(l)].shape))

    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate):
    v_corrected = {}
    s_corrected = {}
    epsilon = 1e-8

    for l in range(1, ((len(parameters)//2)+1)):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1-beta1) * grads['dW' + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1-beta1) * grads['db' + str(l)]
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
        s["dW" + str(l)] = (beta2 * s["dW" + str(l)]) + (1 - beta2) * np.power(grads['dW' + str(l)], 2)
        s["db" + str(l)] = (beta2 * s["db" + str(l)]) + (1 - beta2) * np.power(grads['db' + str(l)], 2)
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * (v_corrected["dW" + str(l)]/ (np.sqrt(s_corrected["dW" + str(l)])+epsilon))
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * (v_corrected["db" + str(l)]/ (np.sqrt(s_corrected["db" + str(l)])+epsilon))
    
    return parameters, v, s, v_corrected, s_corrected

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    sigmoid(z)
    """
     
    return 1 / (1 + np.exp(-z))

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # converting dz to a correct object.
    
    # When z <= 0, should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()

    for l in range(len(parameters)//2):
        parameters["W"+str(l+1)] = params["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = params["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]

    return parameters
    
def predict(X, y, parameters, category, min_confidence=0.5):
    """
    This function is used to predict the results of a L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    
    # Forward propagation
    A, caches, D_list = model_forward(X, parameters, keep_prob=1)

    # convert A to 0/1 predictions
    Y_prediction = (A > min_confidence).astype(int)
    
    #print results
    #print ("predictions: " + str(Y_prediction))
    #print ("true labels: " + str(y))
    print(category, " accuracy: "  + str(np.sum((Y_prediction == y)/m)))
        
    return Y_prediction

def test_image_dl(image_path, parameters, size=64):
    my_label_y = [1] 

    image = np.array(Image.open(image_path).resize((size, size)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, size * size * 3)).T

    my_predicted_image = predict(image, my_label_y, parameters, "Input prediction ")
    classes = np.array(['non-banana', 'banana'])

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + str(classes[int(np.squeeze(my_predicted_image)),]) +  "\" picture.")
    return

def plot_learning_curve(cost, learning_rate, layers_dims):
    """Plot learning curve (with costs)

    Args:
        trained_model (dictionary): must contain the 'costs', 'learning_rate' of a trained model
    """
    costs = np.squeeze(cost)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate) + "/Layers" + str(layers_dims))
    plt.show()