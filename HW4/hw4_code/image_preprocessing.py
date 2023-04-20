from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers
# VS Code was not letting me import the way it was originally implemented, so for local tests I needed those workaround:
# from tensorflow import keras
# from keras import layers

def data_preprocessing(IMG_SIZE=32):
    '''
    In this function you are going to build data preprocessing layers using tf.keras
    First, resize your image to consistent shape
    Second, standardize pixel values to [0,1]
    return tf.keras.Sequential object containing the above mentioned preprocessing layers
    '''
    # HINT :You can resize your images with tf.keras.layers.Resizing,
    # You can rescale pixel values with tf.keras.layers.Rescaling
    
    # # # # print(IMG_SIZE)

    # # # model = tf.keras.models.Sequential([
    # # #     tf.keras.layers.Resizing(height=IMG_SIZE, width=IMG_SIZE),
    # # #     tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    # # #     tf.keras.layers.Rescaling(scale=1./255)
    # # #     ])
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Resizing(height=IMG_SIZE, width=IMG_SIZE))
    model.add(tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)))
    model.add(tf.keras.layers.Rescaling(scale=1./255))
    
    return model

    

def data_augmentation():
    '''
    In this function you are going to build data augmentation layers using tf.keras
    First, add random horizontal flip
    Second, add random rotation with factor of 0.1
    Third, add random zoom (height_factor = -0.2 and width_factor = -0.2)
    return tf.keras.Sequential object containing the above mentioned augmentation layers
    '''
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RandomFlip(mode="horizontal"))
    model.add(tf.keras.layers.RandomRotation(factor=0.1))
    model.add(tf.keras.layers.RandomZoom(-0.2, -0.2))
    
    return model


    

