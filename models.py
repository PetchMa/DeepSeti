import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, ConvLSTM2D
from keras.layers.core import Activation, Flatten
import matplotlib.pyplot as plt
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
from sklearn.utils import shuffle
from keras.losses import binary_crossentropy
from keras.utils import np_utils
import numpy as np
from keras.layers import Reshape
from keras import losses
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, LSTM, MaxPooling1D, Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import Convolution1D
from keras.layers import  Conv2D, MaxPool3D, MaxPooling3D, TimeDistributed, Embedding 
from keras.layers import BatchNormalization

class DeepSeti_Model():
    def __init__(self):
        self.on = True
    def CNNLSTM(dataset_shape, kernel, epoch, learning_rate, cnn_layers, lstm_layers, fully_connected,rate, time):
        LR = LeakyReLU(0.3)
        LR.__name__ = 'relu'
        input_shape = dataset_shape
        kernel = kernel
        epoch = epoch
        learning_rate = learning_rate
        model = Sequential()
        model.add(Convolution1D(32*i, kernel_size = kernel, strides=1, padding='same', input_shape=input_shape ))
        for i in range(1,cnn_layers):
            model.add(Convolution1D(32*i, kernel_size = kernel, strides=1, padding='same'))
            model.add(LR)
            model.add(MaxPooling1D(pool_size=(2)))
            model.add(Dropout(0.5))
        for k in range(1,lstm_layers):
            if k==lstm_layers:
                sequence = False
            else:
                sequence = True
            model.add(CuDNNLSTM(512, return_sequences=sequence))
            model.add(Dropout(0.5))
        for c in range(1,fully_connected)
            model.add(Dense(int(rate*time/c), activation=LR))
            model.add(Dropout(0.5))    
        model.add(Dense(2 , activation='sigmoid'))
        model.summary()
        print(learning_rate/epoch)
        sgd= SGD(lr=learning_rate, decay = learning_rate/epoch, momentum=0.0, nesterov=False)

        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])

        return model
    def CNN(dataset_shape, kernel, epoch, learning_rate, cnn_layers, fully_connected,rate, time):
        LR = LeakyReLU(0.3)
        LR.__name__ = 'relu'
        input_shape = dataset_shape
        kernel = kernel
        epoch = epoch
        learning_rate = learning_rate
        model = Sequential()
        model.add(Convolution1D(32*i, kernel_size = kernel, strides=1, padding='same', input_shape=input_shape ))
        for i in range(1,cnn_layers):
            model.add(Convolution1D(32*i, kernel_size = kernel, strides=1, padding='same'))
            model.add(LR)
            model.add(MaxPooling1D(pool_size=(2)))
            model.add(Dropout(0.5))
        for c in range(1,fully_connected)
            model.add(Dense(int(rate*time/c), activation=LR))
            model.add(Dropout(0.5))

        model.add(Dense(2 , activation='sigmoid'))
        model.summary()
        print(learning_rate/epoch)
        sgd= SGD(lr=learning_rate, decay = learning_rate/epoch, momentum=0.0, nesterov=False)

        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
        return model
    def LSTM(dataset_shape, epoch, learning_rate, lstm_layers, lstm_units, fully_connected,rate, time):
        LR = LeakyReLU(0.3)
        LR.__name__ = 'relu'
        input_shape = dataset_shape
        epoch = epoch
        learning_rate = learning_rate
        model = Sequential()
        model.add(CuDNNLSTM(lstm_units, input_shape =dataset_shape, return_sequences=True))
        for i in range(1, lstm_layers-1):
            model.add(CuDNNLSTM(lstm_units, return_sequences=True))
            model.add(Dropout(0.5))
        model.add(CuDNNLSTM(lstm_units))
        model.add(Dropout(0.5))
        for c in range(1,fully_connected)
            model.add(Dense(int(rate*time/c), activation=LR))
            model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))

        model.summary()
        sgd= SGD(lr=learning_rate, decay = learning_rate/epoch ,momentum=0.0, nesterov=False)

        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])

        return model