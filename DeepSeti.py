from DeepSearch.model import model
from DeepSearch.train import train 
from DeepSearch.classification import classification
from DeepSearch.synthetic import synthetic 
from DeepSearch.classification import classification
from DeepSearch.save_model import save_model 
from DeepSearch.preprocessing import DataProcessing as dp
from keras.models import load_model
from keras.models import Model
from keras.layers import Input

import keras 

class DeepSeti(object):
    def __init__(self):
        self.name="Deep Seti"
    
    def unsupervised_data(self, list_directory):
        dp = dp()
        self.X_train_unsupervised, self.X_test_unsupervised = dp.load_multiple_files(list_directory=list_directory)

    def supervised_data(self, list_directory):
        self.X_train_unsupervised, self.X_test_unsupervised = self.unsupervised_data(list_directory)
        self.X_train_supervised, self.X_test_supervised, self.y_train_supervised, self.y_test_supervised  = synthetic.generate(total_num_samples= 5000, data = X_train_unsupervised)

    def encoder_injection_model_defualt_create(self):
        mod = model(latent_dim=64, kernel_size=(3,3), data_shape=self.X_train_unsupervised[0].shape, layer_filters =[32,64,128], CuDNNLSTM=True)
        self.encode = mod.encoder()
        self.feature_classification = mod.feature_classification()
        self.latent_encode = mod.latent_encode()
        self.decoder = mod.decoder()
        self.inputs = Input(shape=self.X_train_unsupervised[0].shape, name='input')

    def train_custom_data(self):
        train_obj = train()
        train = train_obj.train_model( epoch=5, inputs=self.inputs, encode = self.encode, feature_encode=self.feature_classification, 
                    decoder=self.decoder, latent_encode=self.latent_encode
                    , X_train_unsupervised=self.X_train_unsupervised
                    , X_test_unsupervised=self.X_test_unsupervised
                    , X_train_supervised=self.X_train_supervised
                    , X_test_supervised=self.X_test_supervised
                    , y_train_supervised=self.y_train_supervised,
                    y_test_supervised=self.y_test_supervised)
        save = save_model()
        save.save(train)

    def prediction(self, location, files, anchor):

        predict = predict(anchor = self.X_test_supervised, test=X_test_supervised, model_location="/content/encoder_injected_model (1).h5")
        values = predict.compute_distance()

