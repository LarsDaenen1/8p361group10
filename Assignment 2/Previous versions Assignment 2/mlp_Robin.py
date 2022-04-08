"""
TU/e BME Project Imaging 2021
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard


# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)



# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255


# convert 1D class arrays to 10D class matrices
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

#%% Exersize 1

for Layers in range(1,5):
    for Nodes in [32,64,92,128]:
        model = Sequential()
        # flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
        model.add(Flatten(input_shape=(28,28,1))) 
        # fully connected layer with 64 neurons and ReLU nonlinearity
        for i in range(Layers):
            model.add(Dense(Nodes, activation='relu'))
        # output layer with 10 nodes (one for each class) and softmax nonlinearity
        model.add(Dense(10, activation='softmax')) 


        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # use this variable to name your model
        model_name="Model_" +str(Layers)+'_Layers_'+ str(Nodes)+ '_Nodes'
    
        # create a way to monitor our model in Tensorboard
        tensorboard = TensorBoard("logs/" + model_name)
        
        # train the model
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])
        
        
        score = model.evaluate(X_test, y_test, verbose=0)
        
        
        print("Loss: ",score[0])
        print("Accuracy: ",score[1])
#%% Exersize 2
#2a
model = Sequential()
model.add(Flatten(input_shape=(28,28,1))) 
model.add(Dense(10, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_name="Model_2a"
tensorboard = TensorBoard("logs/" + model_name)
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])
score = model.evaluate(X_test, y_test, verbose=0)
#2b
model = Sequential()
model.add(Flatten(input_shape=(28,28,1))) 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_name="Model_2b"
tensorboard = TensorBoard("logs/" + model_name)
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])
score = model.evaluate(X_test, y_test, verbose=0)
#2c
model = Sequential()
model.add(Flatten(input_shape=(28,28,1))) 
model.add(Dense(64, activation='linear')) # standard
model.add(Dense(64, activation='linear'))
model.add(Dense(64, activation='linear'))
model.add(Dense(10, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_name="Model_2c"
tensorboard = TensorBoard("logs/" + model_name)
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])
score = model.evaluate(X_test, y_test, verbose=0)

#%% Exersize 3
y_train2= np.array([[i[1] or i[7], i[0] or i[6]or i[8] or i[9], i[2] or i[5], i[3] or i[4]] for i in y_train])
y_val2 = np.array([[i[1] or i[7], i[0] or i[6]or i[8] or i[9], i[2] or i[5], i[3] or i[4]] for i in y_val])
y_test2 = np.array([[i[1] or i[7], i[0] or i[6]or i[8] or i[9], i[2] or i[5], i[3] or i[4]] for i in y_test])
model = Sequential()
model.add(Flatten(input_shape=(28,28,1))) 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_name="Model_3"
tensorboard = TensorBoard("logs/" + model_name)
model.fit(X_train, y_train2, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val2), callbacks=[tensorboard])
score = model.evaluate(X_test, y_test2, verbose=0)
