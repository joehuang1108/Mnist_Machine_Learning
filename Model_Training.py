# 1. Get data and resize them to all same size
# 2. Convolution --> activation function + pooling
# 3. Flatten + classify

import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from matplotlib import pyplot

# Loading the training and testing set into separate variables
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 100 --> test and train dataset
# 20-80 --> 80 pics for training, 20 pics for testing
# Feed 80 to train a machine learning model
# Use the 20 to judge how accurate it is

# Print size of the training and testing set
print(train_x.shape) # x is the image
print(train_y.shape) # y is the label to the image (truth)
print(test_x.shape)
print(test_y.shape)

# Display sample dataset
# for i in range(9):
#     pyplot.subplot(i)
#     pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.show()

# input image dimension
img_rows = 28
img_cols = 28

# (60000,28,28) --> (60000,28,28,1)
# Reshaping/cleaning dataset
# M x N x 1
train_x = train_x.reshape(train_x.shape[0], img_rows, img_cols, 1)
test_x = test_x.reshape(test_x.shape[0], img_rows, img_cols, 1)

# train_x /= 255
# test_x /= 255

train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)



# Building model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
# softmax --> allows you to split into multiple categories

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Number of training samples used in one iteration
# 60,000 / 128 = 470
# Takes 470 iterations to go through training dataset
batch_size = 128
# Epoch: Number of cycles going through the training dataset
epochs = 10
model.fit(train_x, train_y,
          batch_size= batch_size,
          epochs= epochs,
          verbose=1,
          validation_data=(test_x,test_y))
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])   # Want as low as possible
print('Test Accuracy:', score[1])   # Want as high as possible
model.save("Test_model.h5")