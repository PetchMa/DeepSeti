 
import tempfile
import pydub
import scipy.io.wavfile
import os, os.path 
import numpy as np

class data:
  def __init__(self, file_path, rate, as_float):
    self.file_path = file_path
    self.rate = rate
    self.as_float = as_float

  def read_wav(self):
    path, ext = os.path.splitext(self.file_path)
    assert ext=='.wav'
    mp3 = pydub.AudioSegment.from_file(self.file_path, format="wav")
    mp3 = mp3.set_frame_rate(self.rate)
    _, path = tempfile.mkstemp()
    mp3.export(path, format="wav")
    rate, data = scipy.io.wavfile.read(path)
    os.remove(path)
    if as_float:
        data = data/(2**15)
    return data

  def count_file(self):
    file_name = self.file_path
    # Searches for all files
    path, dirs, files = next(os.walk(file_name))
    # Finds the number of files. 
    file_count = len(files)
    print('total num of files: ' + str(file_count))
    return file_count

  def load_data(self, time):
    folder = self.file_path
    rate = self.rate
    time_seconds = time
    length_rate = rate*time_seconds
    file_count = count_file()-1
    dataset = np.zeros((file_count, length_rate,2), dtype=float)
    for i in range(0,file_count):
      if i%500==0:
        print(str(i)+" files have been loaded")
      name = folder+'/'+str(i+1)+'.wav'
      dataset[i][:][:] = read_wav()
    print("Tensor shape of loaded dataset is: "+ str(dataset.shape))
    return dataset