import os, os.path
import numpy as np
import tempfile
import pydub
import scipy.io.wavfile

class label():
    def __init__(self):
        self.image = True
    def create_labels(self, dataset, fake):
        labeltrue = np.concatenate((np.ones((dataset.shape[0],1),dtype='int64'),np.zeros((dataset.shape[0],1),dtype='int64')), axis=1)
        labelfalse = np.concatenate((np.zeros((fake.shape[0],1),dtype='int64'),np.ones((fake.shape[0],1),dtype='int64')), axis=1)
        label = np.concatenate((labeltrue, labelfalse))
        print(label.shape)
        return label