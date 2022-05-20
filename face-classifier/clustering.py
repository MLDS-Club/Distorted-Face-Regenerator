# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import decode_predictions 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

from scipy.spatial import distance

class ClusterObject:

    # initialize features
    def __init__(self, data):
        self.data = data
    
    def addFeatureVector(self, filename, age, gender):
        path = r"/Users/arshianayebnazar/Documents/GitHub/Face-Distortion-Fixer/data/crop_part1"
        # change the working directory to the path where the images are located
        os.chdir(path)
        model = VGG16()
        model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
        # load the image as a 224x224 array
        img = load_img(filename, target_size=(224,224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img)
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = img.reshape(1,224,224,3)
        # prepare image for model
        imgx = preprocess_input(reshaped_img)
        # get the feature vector
        feat = model.predict(imgx, use_multiprocessing=True)
        print(decode_predictions(feat))
        # Add feature vector to dictionary
        temp = {}
        for i in self.data.keys():
            if (i[0:2] == str(age-2) or i[0:2] == str(age-1) or i[0:2] == str(age) or i[0:2] == str(age+1) or i[0:2] == str(age+2)) and (i[3] == str(gender)) and (i[5] == '0' or i[5] == '4'):
                temp[i] = self.data[i]
        self.data = temp
        # self.data[filename] = feat
        self.originalImage = feat
    
    def getGroups(self, numOfClusters):
        # get a list of the filenames
        filenames = np.array(list(self.data.keys()))

        # get a list of just the features
        feat = np.array(list(self.data.values()))

        # reshape so that there are 210 samples of 9780 vectors
        feat = feat.reshape(-1,feat.shape[2])

        # reduce the amount of dimensions in the feature vector
        pca = PCA(n_components=100, random_state=22)
        pca.fit(feat)
        x = pca.transform(feat)

        # # cluster feature vectors
        # kmeans = KMeans(n_clusters=numOfClusters, random_state=22)
        # kmeans.fit(x)

        # # holds the cluster id and the images { id: [images] }
        # groups = {}
        # for file, cluster in zip(filenames,kmeans.labels_):
        #     if cluster not in groups.keys():
        #         groups[cluster] = []
        #         groups[cluster].append(file)
        #     else:
        #         groups[cluster].append(file)
        
        # return groups


        # Euclidean distance between feature vectors
        distances = {}
        for i in self.data.keys():
            x = distance.euclidean(self.originalImage, self.data[i])
            distances[i] = x
        return sorted(distances.items(), key=lambda x: x[1])


    
    # function that lets you view a cluster (based on identifier)        
    def view_cluster(self, groups):
        path = r"/Users/arshianayebnazar/Documents/GitHub/Face-Distortion-Fixer/data/crop_part1"
        # change the working directory to the path where the images are located
        os.chdir(path)
        plt.figure(figsize = (25,25))
        # gets the list of filenames for a cluster
        # files = groups[cluster]
        files = [x[0] for x in groups[0:11]]
        # only allow up to 30 images to be shown at a time
        if len(files) > 30:
            print(f"Clipping cluster size from {len(files)} to 30")
            files = files[:29]
        # plot each image in the cluster
        print(files)
        for index, file in enumerate(files):
            plt.subplot(10,10,index+1)
            img = load_img(file)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')
