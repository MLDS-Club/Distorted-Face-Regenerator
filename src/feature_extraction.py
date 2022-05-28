import keras

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import decode_predictions 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# os, numpy, pandas
import os
import numpy as np
import pandas as pd


def getFeatureVector(img):
        # path = r"/Face-Distortion-Fixer/demo"
        # os.chdir(path)
        model = VGG16()
        model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
        img = np.array(img)
        reshaped_img = img.reshape(1,224,224,3)
        imgx = preprocess_input(reshaped_img)
        feat = model.predict(imgx, use_multiprocessing=True)
        print(decode_predictions(feat))
        return feat