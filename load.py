import os, os.path
import numpy as np
import tempfile
import pydub
import scipy.io.wavfile
from read_mp3 import audioRead

class load():
    def __init__(self):
        self.load = True
    def addFiles(self, folder_name, numberFiles, rate, time_seconds):
        folder = folder_name
        file_name = folder_name
        # Searches for all files
        path, dirs, files = next(os.walk(file_name))
        # Finds the number of files. 
        file_count = len(files)
        length_rate = rate*time_seconds
        print('total num of files:' + str(file_count))
        print("DeepSeti Will Load: "+str(numberFiles))
        file_count = numberFiles -1
        dataset = np.zeros((239, length_rate ,2), dtype=float)
        audioRead = audioRead()
        for i in range(0,239):
        if i%numberFiles/10==0:
            print(str(i)+" files have been loaded")
        dataset[i][:][:] = audioRead.read(folder+'/'+str(i+1)+'.wav', rate, True)
        print("Tensor shape: "+ str(dataset.shape))
        return dataset