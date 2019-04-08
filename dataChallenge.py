#!/usr/bin/env python3
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

# import pandas as pd
import numpy as np
# import os
# import re
# import signal
# import sys


model = Sequential()

train_path = "test_x.npy"

test_data = np.load("test_x.npy") #Shape is (200, 168, 308, 3)
train_data = np.load("train_x.npy")
train_label = np.load("train_label.npy")


