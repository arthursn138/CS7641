from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # VS Code was not letting me import the way it was originally implemented, so for local tests I needed those workaround:
# from tensorflow import keras
# from keras import models
# from keras import layers
# from keras import datasets
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
# from keras.layers import LeakyReLU


class CNN(object):
    def __init__(self):
        # change these to appropriate values

        self.batch_size = 64
        self.epochs = 5
        self.init_lr= 1e-3 #learning rate

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(32,32,3)))
        model.add(tf.keras.layers.Conv2D(8, 3, strides=1, padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.Conv2D(32, 3, strides=1, padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Dropout(rate=0.30))
        model.add(tf.keras.layers.Conv2D(32, 3, strides=1, padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.Conv2D(64, 3, strides=1, padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Dropout(rate=0.30))
        model.add(tf.keras.layers.Flatten(input_shape=(None, 8, 8, 64)))
        model.add(tf.keras.layers.Dense(256, activation=LeakyReLU(0.1)))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(128, activation=LeakyReLU(0.1)))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Softmax())

        # for layer in model.layers:
        #     print(layer.output_shape)

        return model
    
    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model and specify the loss, optimizer, and metric.

        The metric we want to use here is CategoricalAccuracy. You can set 
        metrics=['accuracy'] and have tensorflow determine the type of accuracy that is 
        appropriate or you can directly set the metric to [tf.keras.metrics.CategoricalAccuracy()]. 
        Do not define the metric as [tf.keras.metrics.Accuracy()]

        Return: model

        '''
        self.model = self.create_net()
        self.model.compile(optimizer="Adam", loss="CategoricalCrossentropy", metrics=["CategoricalAccuracy"])
        return self.model


        

