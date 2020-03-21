import numpy as np
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, ConvLSTM2D
from keras.layers.core import Activation, Flatten
import matplotlib.pyplot as plt
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
from keras.utils import np_utils
import os, os.path
from scipy.io import wavfile
from keras.models import Model
from keras import backend as K
from random import random
from keras.callbacks import EarlyStopping, ModelCheckpoint



class predict(object):

    def __init__(self, anchor, test, model_location):
        self.anchor = anchor
        self.test = test
        self.encoder_injected = load_model(model_location)
        self.values= np.zeros(self.test.shape[0])
    def compute_distance(self):
        check = self.encoder_injected.predict(self.test)
        anchor = self.encoder_injected.predict(self.anchor)
        for j in range(0, self.test.shape[0]-1):
            index = int(random()*10)
            self.values[j]=(np.square(np.subtract(anchor[index:index+1,:], check[j:j+1,:]))).mean()
        return self.values
    
    def max_index(self, top=3):
        top_hits = []
        copy = self.values
        for i in range(0, top):
            hit = np.argmax(copy)
            top_hits.append(hit)
            copy[hit]=0
        return top_hits