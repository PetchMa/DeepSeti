# DeepSeti - Deep Learning Seti Search Tool
This is a python implementation of DeepSeti - an algorithm designed to detect anomalies for Radio telescope data open sourced by Breakthrough Listen. This module facilitates the custom architecture and training loops required for the DeepSeti algorithm to preform a multichannel search for anomalies. Main objective is to develop software thats that increase the computational sensitivity and speed to search for unpredictable anomalies.  **Rationale:** Currently the code only works for MID-RES filterbank and h5 files. Developments made further will preforming tests on full res products from GBT. 

![alt text](https://github.com/PetchMa/DeepSeti/blob/master/assets/code_block1.png)

# Introduction

The purpose of this algorithm is to help detect anomalies within the GBT dataset from Breakthrough Listen. The code demonstrates a concept that accelerating SETI in large unlabeled datasets. This approach is an extension from the original paper [https://arxiv.org/pdf/1901.04636.pdf] by looking into preforming the final classification on the encoded feature vector by taking a triplet loss between an anchor, positive and negative samples.



# Deep Learning Architecture

This deep learning algorithm uses a novel technique developed for specific SETI use cases. This method basically *injects* an encoder thats been previously trained on a classification dataset into an autoencoder trained through unsupervised techniques. This method relies on a inital small labeled dataset where intermediately trained a CNN-LSTM classifier then injected it into the Auto Encoder. 

**Rationale**: This way we can force the feature selection from CNN's to search for those desired labels while the unsupervised method gives it the “freedom” to familiarize with "normal data" and detect novel anomalies beyond the small labeled dataset. Both the supervised and unsupervised models are executed together and model injections occur intermittently.

*Reference diagram below*

<p align="center"> 
<img src="https://github.com/PetchMa/DeepSeti/blob/master/assets/image%20(3).png">
</p>

# How To Use The Algorithm 

Some features are still under construction, however you can test the current powers of this algorithm following this simple guide below. ** Note: This will require Blimpy and Setigen to opperate properly.** Install these requirements by running the following commands in the terminal in your python enviroment. 

```
pip3 install -r requirements.txt
```

After getting the following setup. Download a radio observation from the UC Berkeley SETI open database. [http://seti.berkeley.edu/opendata]. Or get a test sample by typing this command...
```
wget http://blpd13.ssl.berkeley.edu/dl/GBT_58402_66623_NGC5238_mid.h5
```
Following that all you need to do is clone the repository, and navigate into the folder where its cloned in.
```
git clone https://github.com/PetchMa/DeepSeti.git
```

Once you're within the cloned folder, copy the code block into a new python script. Fill in the mising directories, and you can train a model on your custom data. 


```python

direct = ['/-Your directory-/data.h5',
          '/-Your directory-/data2.h5',
          '/-Your directory-/data3.h5'
          ]

DeepSeti = DeepSeti()
DeepSeti.unsupervised_data(direct)

DeepSeti.supervised_data(direct)

DeepSeti.encoder_injection_model_defualt_create()

DeepSeti.train_custom_data()
DeepSeti.prediction(model_location="model.h5", test_location="data1.h5", 
                    anchor_location="data2.h5", top_hits=4)

```

<p align="center"> 
<img src="https://github.com/PetchMa/DeepSeti/blob/master/assets/image%20(4).png">
</p>
