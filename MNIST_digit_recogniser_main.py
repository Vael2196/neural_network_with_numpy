#Author: Vadim Filyakin
#Last-modified: 08/08/24

import numpy as np
import pandas as pnd
from matplotlib import pyplot as ppl

#The dataset is taken from the following link:
#https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer?resource=download
#published under CC0: Public Domain

data = pnd.read_csv("Dataset/train.csv")
data.head()

data = np.array(data)
m, n = data.shape #dimensions of the data, where m is the number of rows and n is the amount of features +1
np.random.shuffle(data)

data_dev = data[0:1000].T #data transposition
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255.

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.
_,m_train = x_train.shape


def init_params():
    "initialises parameters for Bayes' theorem"
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1, b1, W2, b2

def ReLU(z):
    "Activation function for layer [1]. Returns x if x>0, otherwise return 0"
    return np.maximum(z, 0)

def ReLu_derivative(z):
    "Calculates slope (It can be either 0 if x is negative, or 1 if its positive)"
    return z>0

def softmax(z):
    "Calculates probability"
    return np.exp(z) / sum(np.exp(z))

def forward_prop(W1, b1, W2, b2, x):
    "Forward propagetion function"
    Z1 = W1.dot(x) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_col_matrix(y):
    "Creates a matrix with one at the position of the predicted number"
    one_col_y = np.zeros((y.size, y.max() + 1))
    one_col_y[np.arange(y.size), y] = 1
    one_col_y = one_col_y.T
    return one_col_y

def backprop(Z1, A1, Z2, A2, W1, W2, x, y):
    "Backward propagation of errors"
    one_col_y = one_col_matrix(y)
    dZ2 = A2 - one_col_y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLu_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(x.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    "Update parameters with a considiration of error propagation"
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    "Getter for an array of data"
    return np.argmax(A2, 0)

def get_accuracy(predictions, y):
    "Getter for the accuracy of predictions of the network"
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, alpha, iterations):
    "Loops through the data and prints out the result and deviation for each loop with a gap of alpha (e.g. alpha = 0.1 means every 10th loop gets printed)"
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x)
        dW1, db1, dW2, db2 = backprop(Z1, A1, Z2, A2, W1, W2, x, y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, 500)

def make_predictions(x, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, x)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    ppl.gray()
    ppl.imshow(current_image, interpolation='nearest')
    ppl.show()

test_prediction(6, W1, b1, W2, b2)
# dev_predictions = make_predictions(x_dev, W1, b1, W2, b2)
# get_accuracy(dev_predictions, y_dev)