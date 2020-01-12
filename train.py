import import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, ConvLSTM2D
from keras.layers.core import Activation, Flatten
import matplotlib.pyplot as plt
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
from sklearn.utils import shuffle
from keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import os, os.path
import numpy as np

from keras.layers import Reshape
from keras import losses
from keras.layers.advanced_activations import LeakyReLU
from pydub import AudioSegment
from scipy.io import wavfile
from keras.layers import Input, LSTM, MaxPooling1D, Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import Convolution1D
from keras.layers import  Conv2D, MaxPool3D, MaxPooling3D, TimeDistributed, Embedding 
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model

class train():
    def __init__(self):
        self.on =True
    def model_train(self, model,X_train,y_train,X_test, y_test,  batch_size, epoch, file_name):
        batch_size =batch_size
        # es_callback = EarlyStopping(monitor='val_loss', mode='min')
        mc = ModelCheckpoint(file_name+'.h5', monitor='val_loss', mode='min', save_best_only=True)
        history = model.fit(X_train, y_train,  batch_size=batch_size, epochs=epoch, validation_data=(X_test, y_test), callbacks=[mc])
        return history
    def plot_graph(self, history):
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left') 
        plt.show()
