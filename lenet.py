from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras.models import Model
from keras import layers
from keras import backend as K
import tensorflow as tf


class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, (5, 5,), padding="same",
            input_shape=(height, width,depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

class functional_Lenet():
    @staticmethod
    def build(width, height, depth, classes, bn=True):
        input_ayer = layers.Input(shape=(height, width,depth))
        conv1 = layers.Conv2D(20, (5, 5,), padding="same",activation='relu')(input_ayer)
        pool1 = layers.MaxPool2D((2,2),strides=(2,2))(conv1)
        if bn:
            pool1 = layers.Lambda(functional_Lenet.batchnorm)(pool1)
        #pool1 = layers.BatchNormalization()(pool1)
        conv2 = layers.Conv2D(50, (5, 5), padding="same")(pool1)
        pool2 = layers.MaxPool2D((2,2),strides=(2,2))(conv2)
        if bn:
            pool2 = layers.Lambda(functional_Lenet.batchnorm)(pool2)
        #pool2 = layers.BatchNormalization()(pool2)

        fc_input = layers.Flatten()(pool2)
        fc_output = layers.Dense(500)(fc_input)
        if bn:
            fc_output = layers.Lambda(functional_Lenet.batchnorm)(fc_output)
        #fc_output = layers.BatchNormalization()(fc_output)

        probabilites = layers.Dense(classes,activation='softmax')(fc_output)
        model = Model(inputs=input_ayer, outputs=probabilites)
        return model


    @staticmethod
    def batchnorm(pool1):
        pool1_avg = tf.math.reduce_mean(pool1,axis=0,keepdims=True)
        pool1_std = tf.math.reduce_mean(tf.math.pow(pool1-pool1_avg,2),axis=0,keepdims=True)
        denum = 1/tf.math.sqrt(pool1_std+1e-5)
        return tf.math.multiply (pool1-pool1_avg,denum)



