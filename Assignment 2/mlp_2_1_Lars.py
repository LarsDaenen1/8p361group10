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

file_out = r"C:\Users\larsd\OneDrive - TU Eindhoven\Universiteit\Jaar 3\Kwartiel 3\8P361 - Project Imaging\8p361-project-imaging-master\assignments\Experiments assignment 2.1.txt"
#experiments = open(file_out, "w")


# fully connected layers with 64 neurons and ReLU nonlinearity
import seaborn as sns
num_layers = [1, 2, 4, 8, 16]
num_neurons = [32, 64, 128, 256]
experiments_acc = []
experiments_loss = []
for i in num_layers:
    layer_acc = []
    layer_loss = []
    for j in num_neurons:
        model = Sequential()
        # flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
        model.add(Flatten(input_shape=(28,28,1))) 
        for k in range(i):
            model.add(Dense(j, activation='relu'))

        # output layer with 10 nodes (one for each class) and softmax nonlinearity
        model.add(Dense(10, activation='softmax')) 

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # use this variable to name your model
        model_name= str("ReLU_activation_"+str(j)+"neurons_"+str(i)+"hidden_layers")

        # create a way to monitor our model in Tensorboard
        tensorboard = TensorBoard("logs/" + model_name)

        # train the model
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])

        score = model.evaluate(X_test, y_test, verbose=0)

        print("Loss: ",score[0])
        print("Accuracy: ",score[1])
        layer_loss.append(score[0])
        layer_acc.append(score[1])
        #experiment = "# layers: {}    # neurons: {}    Loss: {:.3f}, Accuracy: {:.3f}\n"
        #experiments.write(experiment.format(i, j, score[0], score[1]))
    experiments_acc.append(layer_acc)
    experiments_loss.append(layer_loss)
#experiments.close()                 

fig = plt.figure()
ax = sns.heatmap(experiments_acc, annot=True, cmap="RdYlGn", fmt='.3g', 
                 xticklabels=num_neurons, yticklabels=num_layers)
ax.set(xlabel='# Neurons', ylabel='# Layers')
ax.set_title("Accuracy")
fig = plt.figure()
ax = sns.heatmap(experiments_loss, annot=True, cmap="RdYlGn_r", fmt='.3g', 
                 xticklabels=num_neurons, yticklabels=num_layers)
ax.set(xlabel='# Neurons', ylabel='# Layers')
ax.set_title("Loss")
print('\a')
