from models import DeepSeti_Model
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


class DeepSeti():
    def __init__(self):
        self.on = True
    def BuildModel_CNNLSTM(self, dataset_shape, kernel, epoch, learning_rate, 
        cnn_layers, lstm_layers, fully_connected,rate, time):
        DeepSeti_Model = DeepSeti_Model()
        model = Sequential()
        model = DeepSeti_Model.CNNLSTM(dataset_shape, kernel, epoch, learning_rate, cnn_layers, lstm_layers, fully_connected,rate, time)
        return model
    def BuildModel_LSTM(self, dataset_shape, kernel, epoch, learning_rate, cnn_layers, fully_connected,rate, time):
        DeepSeti_Model = DeepSeti_Model()
        model = Sequential()
        model = DeepSeti_Model.LSTM( dataset_shape, epoch, learning_rate, lstm_layers, lstm_units, fully_connected,rate, time)
        return model
    def BuildModel_CNN(self, dataset_shape, epoch, learning_rate, lstm_layers, lstm_units, fully_connected,rate, time)
        DeepSeti_Model = DeepSeti_Model()
        model = Sequential()
        model = DeepSeti_Model.CNNLSTM(dataset_shape, kernel, epoch, learning_rate, cnn_layers, lstm_layers, fully_connected,rate, time)
        return model
    
    
    