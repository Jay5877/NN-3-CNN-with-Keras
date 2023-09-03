# Shah, Jay Bijal
# 1002_070_971
# 2023_04_01
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, InputLayer, Convolution2D, MaxPooling2D

def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(y_pred)):
      cm[y_true[i],y_pred[i]]+=1  
    return cm

def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    tf.keras.utils.set_random_seed(5368) # do not remove this line
    model = Sequential()
    model.add(Convolution2D(8,(3,3), strides = (1,1), padding= 'same',activation="relu", input_shape =(28, 28, 1), kernel_regularizer= tf.keras.regularizers.L2(0.0001)))
    model.add(Convolution2D(16,(3,3), strides = (1,1), padding= 'same',activation="relu", kernel_regularizer= tf.keras.regularizers.L2(0.0001)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(32,3, strides = (1,1), padding= 'same',activation="relu", kernel_regularizer= tf.keras.regularizers.L2(0.0001)))
    model.add(Convolution2D(64,3, strides = (1,1), padding= 'same',activation="relu", kernel_regularizer= tf.keras.regularizers.L2(0.0001)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation= "relu", kernel_regularizer= tf.keras.regularizers.L2(0.0001)))
    model.add(Dense(10, activation= "linear", kernel_regularizer= tf.keras.regularizers.L2(0.0001)))
    model.add(Activation('softmax'))
    model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
    model.save('model.h5')
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.2)
    output = model.predict(X_test)
    output = np.argmax(output, axis = 1)
    Y_test = np.argmax(Y_test, axis = 1)
    cm = confusion_matrix(Y_test, output)
    plt.matshow(cm)
    plt.savefig('confusion_matrix.png')
    model_load = keras.models.load_model('model.h5')
    answer = []
    answer.append(model_load)
    answer.append(history)
    answer.append(cm)
    answer.append(np.asarray(output))
    return answer