import matplotlib.pyplot as plt
import seaborn as sns 
import keras 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2 
import os 
import numpy as np

titles = ['with_mask', 'without_mask']
img_size = 224
def retData(dataDir):
    data = []
    for title in titles:
        path = os.path.join(dataDir, title)
        classNum = titles.index(title)
        for img in os.listdir(path):
            try:
                imgArr = cv2.imread(os.path.join(path,img))[...,::-1]
                resizedArr = cv2.resize(imgArr,(img_size, img_size))
                data.append([resizedArr, classNum])
            except Exception as e:
                print(e)
    return np.array(data)
train = retData('with_mask')
val = retData('without_mask')