import keras
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
import numpy as np
from keras.layers import Reshape
from keras import losses
from keras.layers.advanced_activations import LeakyReLU 
from keras.activations import sigmoid
from keras.layers import Input, LSTM, MaxPooling1D, Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import Convolution1D
from keras.layers import  Conv2D, MaxPool3D, MaxPooling2D, TimeDistributed, Embedding, Convolution2D , Lambda
from keras.layers import BatchNormalization

from blimpy import Waterfall
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization, ZeroPadding2D
from keras.layers import Softmax
from  keras.backend import expand_dims


class model(object):
    def __init__(self, latent_dim, kernel_size, data_shape):
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.data_shape = data_shape

    def encoder(self, data_shape):
        layer_filters = [32,64,128]
        latent_dim = self.latent_dim
        time = int(data_shape)
        filters = layer_filters[1]*2

        inputs = Input(shape=self.data_shape, name='encoder_input')
        x = inputs
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding='same', kernel_initializer='glorot_normal')(x)
            # x = MaxPooling2D(2)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=128,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding='same', kernel_initializer='glorot_normal')(x)
        # x = MaxPooling2D(8)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((32, 128))(x)
        shape = K.int_shape(x)
        x = CuDNNLSTM(32, return_sequences=True, input_shape=(shape))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = CuDNNLSTM(32, return_sequences=True, input_shape=(shape))(x)
        x = LeakyReLU(alpha=0.2)(x)
        self.shape = K.int_shape(x)
        encoder = Model(inputs, x, name='encoder')
        return encoder
        
    def feature_classification(self):
        latent_inputs = Input(shape=(self.shape[1],self.shape[2]), name='fully_connected_inputs')
        x = Flatten()(latent_inputs)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(2)(x)
        x = Softmax()(x)
        fully_connected = Model(latent_inputs, x, name='encoder')
        return fully_connected

    def latent_encode(self, latent_dim):
        latent_inputs = Input(shape=(self.shape[1],self.shape[2]), name='fully_connected_inputs')
        x = Flatten()(latent_inputs)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(latent_dim)(x)
        x = LeakyReLU(alpha=0.2)(x)
        feature_encode = Model(latent_inputs, x, name='encoder')

    def decoder(self):
        layer_filters = [32, 64, 128]
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(64)(latent_inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(shape[1] * shape[2])(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((shape[1], shape[2]))(x)
        x = CuDNNLSTM(32, return_sequences=True, input_shape=(shape))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = CuDNNLSTM(32, return_sequences=True, input_shape=(shape))(x)
        x = LeakyReLU(alpha=0.2)(x)
        shape_1 = K.int_shape(x)
        x = Reshape((self.shape_1[1], 1,  self.shape_1[2]))(x)
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,kernel_size=self.kernel_size, strides=1, padding='same' ,kernel_initializer='glorot_normal')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, x, name='decoder')
        return decoder
