import numpy as np
import pandas as pnd
from matplotlib import pyplot as ppl

#The dataset is taken from the following link:
#https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer?resource=download
#CC0: Public Domain

data = pnd.read_csv("Dataset/train.csv")
data.head()

data = np.array(data)
m, n = data.shape #dimensions of the data, where m is the number of rows and n is the amount of features +1
np.random.shuffle(data)

data_dev = data[0:1000].T #data transposition

y_dev = data_dev[0]
x_dev = data_dev[1:n]

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]

print(x_train[:, 0].shape) #test the first column