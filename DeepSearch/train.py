import keras
from keras.models import Sequential 
from keras.layers.core import Activation, Flatten
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
from keras.losses import binary_crossentropy
from keras.utils import np_utils
import numpy as np
from keras import losses
from keras.models import Model
from keras import backend as K
from  keras.backend import expand_dims
from keras.callbacks import EarlyStopping, ModelCheckpoint

class train(object):
    def __init__(self, epoch, encoder_final, AutoEncoder, X_train_unsupervised, X_test_unsupervised, X_train_supervised, X_test_supervised, y_train_supervised, y_test_supervised, batch_size=4096):
        self.epoch = epoch
        self.X_train_unsupervised =X_train_unsupervised
        self.X_test_unsupervised = X_test_unsupervised
        self.X_train_supervised = X_train_supervised
        self.X_test_supervised = X_test_supervised
        self.y_train_supervised = y_train_supervised 
        self.y_test_supervised = y_test_supervised
        self.encoder_final = encoder_final
        self.AutoEncoder = AutoEncoder
        self.batch_size = batch_size

    def training(self):
        history_encoder_tracker = np.zeros((self.epoch))
        history_unsupervised_tracker = np.zeros((self.epoch))
        sgd_encoder = SGD(lr=0.1, clipnorm=1, clipvalue=0.5)
        sgd_unsupervised = SGD(lr=0.1, clipnorm=1, clipvalue=0.5)
        self.encoder_final.compile(loss='binary_crossentropy', optimizer=sgd_encoder,  metrics=['acc'])
        self.AutoEncoder.compile(loss='mean_squared_error', optimizer=sgd_unsupervised,  metrics=['acc'])

        for i in range(0,self.epoch):
        
        # encoder_final.compile(loss='mean_squared_error', optimizer=sgd_encoder,  metrics=['acc'])
            if i%10==0:
                print("--------------ENCODER TRAIN--------------" + str(i))
                mc = ModelCheckpoint('encoder_model.h5', monitor='val_loss', mode='min', save_best_only=True)
                history_encoder = self.encoder_final.fit(self.X_train_supervised, self.y_train_supervised, batch_size=512, epochs=40, 
                    validation_data=(self.X_test_supervised, self.y_test_supervised), callbacks=[mc])
                print()
                print()
        
        # history_encoder_tracker[i] = history_encoder.history['loss']
        print("--------------UNSUPERVISED TRAIN--------------" + str(i))
        # AutoEncoder.compile(loss='mean_squared_error', optimizer=sgd_unsupervised,  metrics=['acc'])

        mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)
        history_unsupervised = self.AutoEncoder.fit(self.X_train_unsupervised, self.X_train_unsupervised,  batch_size=4096, epochs=1, 
            validation_data=(self.X_test_unsupervised, self.X_test_unsupervised), callbacks=[mc])
        # history_unsupervised_tracker[i] = history_unsupervised.history['loss']
        
