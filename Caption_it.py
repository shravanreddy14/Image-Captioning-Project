#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Dropout,Dense,Embedding,LSTM
from keras.layers.merge import add
import collections


# In[6]:


model = load_model("./model_weights/model_1.h5")


# In[15]:


model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[16]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)


# In[17]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    #Normalization of the preprocessed image.
    img = preprocess_input(img)
    return img


# In[18]:


def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    #print(feature_vector.shape)
    return feature_vector


# In[19]:


with open("word2idx.pkl","rb") as f:
    word2idx = pickle.load(f)
with open("idx2word.pkl","rb") as f:
    idx2word = pickle.load(f)


# In[20]:


def predict_captions(testImg):
    input_text = "startseq"
    maxLen = 35
    for i in range(maxLen):
        seq = [word2idx[w] for w in input_text.split() if w in word2idx]
        seq = pad_sequences([seq],maxlen=maxLen,padding='post')
        
        yPred = model.predict([testImg,seq])
        yPred = yPred.argmax()
        word = idx2word[yPred]
        input_text += (' ' + word)
        
        if word == 'endseq':
            break
            
    output_caption = input_text.split()[1:-1]
    output_caption = ' '.join(output_caption)
    
    return output_caption


# In[21]:

def caption_this_image(image):

    enc = encode_img(image)
    cap = predict_captions(enc)
    
    return cap


# In[ ]:





# In[ ]:




