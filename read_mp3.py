import os, os.path
import numpy as np
import tempfile
import pydub
import scipy.io.wavfile

class audioRead():
    def __init__(self):
        self.on  = True
    def read(file_path, rate,  as_float = False):
        path, ext = os.path.splitext(file_path)
        assert ext=='.wav'
        mp3 = pydub.AudioSegment.from_file(file_path, format="wav")
        mp3 = mp3.set_frame_rate(rate)
        _, path = tempfile.mkstemp()
        mp3.export(path, format="wav")
        rate, data = scipy.io.wavfile.read(path)
        os.remove(path)
        if as_float:
            data = data/(2**15)
        return data