import tensorflow as tf 
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
import tempfile
import pydub
import scipy.io.wavfile
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

class test():
    def __init__(self):
        self.set = True
    def test_execute(self,model, test_true, test_false):
        labeltrue_test = np.concatenate((np.ones((test_true.shape[0],1),dtype='int64'),np.zeros((test_true.shape[0],1),dtype='int64')), axis=1)
        print(labeltrue_test.shape)
        labelfalse_test = np.concatenate((np.zeros((test_false.shape[0],1),dtype='int64'),np.ones((test_false.shape[0],1),dtype='int64')), axis=1)
        print(labelfalse_test.shape)
        test_label = np.concatenate((labeltrue_test, labelfalse_test))
        test = np.concatenate((test_true, test_false))
        print(np.argmax(test_label, axis=1))
        print(np.argmax(model.predict(test), axis=1))
        cm=confusion_matrix(np.argmax(test_label, axis=1), np.argmax(model.predict(test), axis=1))
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()