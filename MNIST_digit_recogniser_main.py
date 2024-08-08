import numpy as np
import pandas as pnd
from matplotlib import pyplot as ppl

#The dataset is taken from the following link:
#https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer?resource=download
#CC0: Public Domain

data = pnd.read_csv("Dataset/train.csv")
data.head()