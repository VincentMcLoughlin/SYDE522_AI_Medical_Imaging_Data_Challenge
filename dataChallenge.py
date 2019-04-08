#!/usr/bin/env python3
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# import pandas as pd
import numpy as np
# import os
# import re
# import signal
# import sys


model = Sequential()

train_path = "test_x.npy"

test_images = np.load("test_x.npy") #Shape is (200, 168, 308, 3), results will be output to the csv
x_data = np.load("train_x.npy")
y_data = np.load("train_label.npy") #shape is (760,)

#Split x_data and y_data into 150 validation images with the rest as training images
x_train = x_data[:500,:,:]
print(len(x_data)-len(x_train))
x_test = x_data[:(-1*(len(x_data)-len(x_train)))]


y_train = y_data[:500]
y_test = y_data[:(-1*(len(x_data)-len(x_train)))]
print(len(test_images))
# Seems to be a collection of images, where each image has the form


# [[223 151 228]
#    [225 144 230]
#    [220 119 217]
#    ...
#    [249 244 252]
#    [248 232 249]
#    [235 190 232]]]

num_classes=20
input_shape = (200,168,308)
model.add(Conv2D(32, kernel_size=(10,10), activation='relu',input_shape=input_shape)) #32 is Number of output channels, number is picked at random and probably incorrect
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2))) #20 classes
model.add(Flatten())
model.add(Dense(units=1000, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=168,
          epochs=3,
          #verbose=1,
          validation_data=(x_test, y_test),
          )

model.save('imageID.neural_net')