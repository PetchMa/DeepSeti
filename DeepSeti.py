from DeepSeti_utils.model import model
from DeepSeti_utils.train import train as training
from DeepSeti_utils.synthetic import synthetic 
from DeepSeti_utils.predict import predict as prediction_algo
from DeepSeti_utils.save_model import save_model 
from DeepSeti_utils.preprocessing import DataProcessing as DataProcessing
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
import pylab as plt
import keras 
from keras.models import load_model
import numpy as np
import time as time 

class DeepSeti(object):
    def __init__(self):
        self.name="Deep Seti"
    
    def unsupervised_data(self, list_directory):
        dp = DataProcessing()
        self.X_train_unsupervised, self.X_test_unsupervised = dp.load_multiple_files(list_directory=list_directory)

    def supervised_data(self, list_directory):
        synth = synthetic()
        self.X_train_supervised, self.X_test_supervised, self.y_train_supervised, self.y_test_supervised  = synth.generate(total_num_samples= 5000, 
                                                                                                                                data = self.X_train_unsupervised[0:10000,:,:,:])
    def encoder_injection_model_defualt_create(self, CuDNNLSTM):
        mod = model(latent_dim=64, kernel_size=(3,3), data_shape=self.X_train_unsupervised[0].shape, layer_filters =[32,64,128], CuDNNLSTM=CuDNNLSTM)
        self.encode = mod.encoder()
        self.feature_classification = mod.feature_classification()
        self.latent_encode = mod.latent_encode()
        self.decoder = mod.decoder()
        self.inputs = Input(shape=self.X_train_unsupervised[0].shape, name='input')

    def train_custom_data(self, epoch, batch, save_file=True):
        train_obj = training()
        train = train_obj.train_model( epoch=epoch, inputs=self.inputs, encode = self.encode, 
                    feature_encode=self.feature_classification, 
                    decoder=self.decoder, latent_encode=self.latent_encode
                    , X_train_unsupervised=self.X_train_unsupervised
                    , X_test_unsupervised=self.X_test_unsupervised
                    , X_train_supervised=self.X_train_supervised
                    , X_test_supervised=self.X_test_supervised
                    , y_train_supervised=self.y_train_supervised
                    , y_test_supervised=self.y_test_supervised
                    , batch_size=batch)
        if save_file:
            save = save_model()
            save.save(train)
    
    def load_model_function(self, model_location):
        self.model_loaded = load_model(model_location)
    
    def convert_np_to_mhz(self, np_index, f_stop,f_start, n_chans):
        width = (f_stop-f_start)/n_chans
        return width*np_index + f_stop

    def prediction(self, test_location, anchor_location, top_hits, target_name, output_folder):
        
        dp_1 = DataProcessing()
        anchor = dp_1.load_data(anchor_location)
        dp = DataProcessing()
        self.test = dp.load_data(test_location)
        f_stop = dp.f_stop
        f_start = dp.f_start
        n_chan =dp.n_chans
        start_time = time.time()
        predict = prediction_algo(anchor = anchor , test=self.test, model_loaded=self.model_loaded)
        self.values = predict.compute_distance()
        self.hits = predict.max_index(top_hits)
        
        fig = plt.figure(figsize=(20, 6))
        plt.plot(self.values)
        plt.xlabel("Number Of Samples")
        plt.ylabel("Euclidean Distance")
        
        for i in range(0,top_hits):
            fig = plt.figure(figsize=(10, 6))
            plt.title('')
            plt.imshow(self.test[self.hits[i],:,0,:], aspect='auto')
            plt.xlabel("fchans")
            plt.ylabel("Time")
            plt.colorbar()
            np_index_start = int(self.hits[i]*4)-16
            np_index_end = int(self.hits[i]*4)+16
            freq_start = self.convert_np_to_mhz(np_index =np_index_start , f_stop=f_stop,f_start=f_start, n_chans=n_chan)
            freq_end = self.convert_np_to_mhz(np_index =np_index_end , f_stop=f_stop,f_start=f_start, n_chans=n_chan)

            np.save(output_folder+"numpy_"+str(target_name.replace('mid.h5','_mid_h5_'))+"index_"+str(self.hits[i])+"_hit_"+str(i)+".npy", self.test[self.hits[i],:,:,:]) 

            plt.title(str(target_name.replace('mid.h5','_mid_h5_'))+"npIndex_"+str(np_index_start)+"_Freq_range_"+str(round(freq_start,4))+'-'+str(round(freq_end,4))+"_hit_"+str(i))
            fig.savefig(output_folder+"image_"+str(target_name.replace('mid.h5','_mid_h5_'))+"Freq_range_"+str(round(freq_start,4))+'-'+str(round(freq_end,4))+"_hit_"+str(i)+".PNG", bbox_inches='tight')
        delta_time = time.time()- start_time
        print("Search time [s]:"+str(delta_time))
    
    def prediction_numpy(self, numpy_data, list_names, anchor_location, top_hits, target_name, output_folder):
        dp_1 = DataProcessing()
        anchor = dp_1.load_data(anchor_location)
        dp = DataProcessing()
        self.test = numpy_data
        start_time = time.time()
        predict = prediction_algo(anchor = anchor , test=self.test, model_loaded=self.model_loaded)
        self.values = predict.compute_distance()
        self.hits = predict.max_index(top_hits)
        
        fig = plt.figure(figsize=(20, 6))
        plt.plot(self.values)
        plt.xlabel("Number Of Samples")
        plt.ylabel("Euclidean Distance")
        
        for i in range(0,top_hits):
            fig = plt.figure(figsize=(10, 6))
            plt.title('')
            plt.imshow(self.test[self.hits[i],:,0,:], aspect='auto')
            plt.xlabel("fchans")
            plt.ylabel("Time")
            plt.colorbar()
            np.save(output_folder+list_names[self.hits[i]], self.test[self.hits[i],:,:,:]) 
            plt.title(str(list_names[self.hits[i]]))
            fig.savefig(output_folder+list_names[self.hits[i]]+"PNG", bbox_inches='tight')
        delta_time = time.time()- start_time
        print("Search time [s]:"+str(delta_time))


