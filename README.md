# DeepSeti
This is a python implementation of DeepSeti - an algorithm designed to detect anomalies for Radio telescope data open sourced by Breakthrough Listen. This module facilitates the custom architecture and training loops required for the DeepSeti algorithm to preform a multichannel search for anomalies. Main objective is to develop software thats that increase the computational sensitivity and speed to search for unpredictable anomalies.  

![alt text](https://github.com/PetchMa/DeepSeti/blob/master/assets/code_block1.png)

#Introduction

The purpose of this algorithm is to help detect anomalies within the GBT dataset from Breakthrough Listen. The code demonstrates a concept that accelerating SETI in large unlabeled datasets. This approach is an extension from the original paper [https://arxiv.org/pdf/1901.04636.pdf] by looking into preforming the final classification on the encoded feature vector by taking a triplet loss between an anchor, positive and negative samples.

#Deep Learning Architecture 
This deep learning algorithm uses a novel technique developed for specific SETI use cases. This method basically *injects* an encoder thats been previously trained on a classification dataset into an autoencoder trained through unsupervised techniques. This method relies on a inital small labeled dataset where intermediately trained a CNN-LSTM classifier then injected it into the Auto Encoder. 

**Rationale**: This way we can force the feature selection from CNN's to search for those desired labels while the unsupervised method gives it the “freedom” to familiarize with "normal data" and detect novel anomalies beyond the small labeled dataset. Both the supervised and unsupervised models are executed together and model injections occur intermittently.

*Reference diagram below*



![alt text](https://github.com/PetchMa/DeepSeti/blob/master/assets/image%20(3).png)

