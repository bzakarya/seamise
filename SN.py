# !pip install opencv-python
# !pip install matplotlib
# !pip install numpy

import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# !pip install tensorflow

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

def first_part_layer():
    inp = Input(shape=(105,105,3), name='input image')
    
    # block 1
    c1 = Conv2D(64, (10,10), activation='relu', name='c1')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same', name='m1')(c1)
                      
    # block 2            
    c2 = Conv2D(128, (7,7), activation='relu', name='c2')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same', name='m2')(c2)
                      
    # block 3
    c3 = Conv2D(128, (4,4), activation='relu', name='c3')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same', name='m3')(c3)
                      
    # block 4
    c4 = Conv2D(256, (4,4), activation='relu', name='c4')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid', name='d1')(f1)
                      
    return Model(inputs=[inp],outputs=[d1] ,name='make_first_layer')

first_part = first_part_layer()
first_part.summary()

# didtatnce class
class L1Dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()
    def call(self, input_mode, validation_mode):
        return tf.math.abs(input_mode - validation_mode)


def seamise_model():
    #
    input_image = Input(shape=(105,105,3), name='input image')
    
    #
    validation_image = Input(shape=(105,105,3), name='validation image')
    
    # combine
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(first_part(input_image), first_part(validation_image))
    
    # add dence layer 
    
    classifier = Dense(1, activation = 'sigmoid')(distances)
    
    return  Model(inputs=[input_image, validation_image] ,outputs = classifier, name='SiameseNetwork')

Siamese_Network = seamise_model()

Siamese_Network.summary()
    