# implementing a new model, this time with keras to overcome the bug

from __future__ import print_function
from laedr import modeleag as laedrM
from laedr import network as laedrNetwork
import sklearn
import tensorflow as tf
import numpy as np
import sys
import os

pretrainedLaedrModelPath = './laedr/model/'

class LAEDR_AIM(laedrM.Model):
    def initialize(self):
        self.encoder = laedrNetwork.EncoderNet()
        optim_default = tf.compat.v1.train.AdamOptimizer(0.0001)
        saver = laedrM.Saver(self, optim_default)
        saver.restore(pretrainedLaedrModelPath)



LAEDR_model = None


def create_model_v3():
    global LAEDR_model
    LAEDR_model = LAEDR_AIM()

    input_mothers = tf.keras.Input(shape=(128, 128, 3))
    input_fathers = tf.keras.Input(shape=(128, 128, 3))
    input_children = tf.keras.Input(shape=(128, 128, 3))


    mother_aif = LAEDR_model.encoder(input_mothers)
    father_aif = LAEDR_model.encoder(input_fathers)
    child_aif = LAEDR_model.encoder(input_children)



    # cm net: concat child and mother, and add dense layers
    cm_concat = tf.keras.layers.concatenate([child_aif, mother_aif], axis=1)

    cm_layer = tf.keras.layers.Dense(90, activation=tf.nn.sigmoid, use_bias = False)(cm_concat)
    cm_layer = tf.keras.layers.Dense(80, activation=tf.nn.sigmoid, use_bias = False)(cm_layer)
    cm_layer = tf.keras.layers.Dense(70, activation=tf.nn.sigmoid, use_bias = False)(cm_layer)
    cm_layer = tf.keras.layers.Dense(60, activation=tf.nn.sigmoid, use_bias = False)(cm_layer)


    # cf net: concat child and father, and add dense layers
    cf_concat = tf.keras.layers.concatenate([child_aif, father_aif], axis=1)
    cf_layer = tf.keras.layers.Dense(90, activation=tf.nn.sigmoid, use_bias = False)(cf_concat)
    cf_layer = tf.keras.layers.Dense(80, activation=tf.nn.sigmoid, use_bias = False)(cf_concat)
    cf_layer = tf.keras.layers.Dense(70, activation=tf.nn.sigmoid, use_bias = False)(cf_concat)
    cf_layer = tf.keras.layers.Dense(60, activation=tf.nn.sigmoid, use_bias = False)(cf_concat)



    # merge cm and cf nets. create final layer
    cf_cm_concat = tf.keras.layers.concatenate([cf_layer, cm_layer], axis=1)
    final_layer = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid, use_bias=True)(cf_cm_concat)
    final_layer = tf.keras.layers.Dense(75, activation=tf.nn.sigmoid, use_bias=True)(final_layer)
    final_layer = tf.keras.layers.Dense(50, activation=tf.nn.sigmoid, use_bias=True)(final_layer)
    final_layer = tf.keras.layers.Dense(25, activation=tf.nn.sigmoid, use_bias=True)(final_layer)
    final_layer = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid, use_bias=True)(final_layer)
    final_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True)(final_layer)
    model_output = final_layer


    model = tf.keras.Model(inputs=[input_fathers, input_mothers, input_children], outputs=model_output) 
    return model

# shuffles the batches such that we have a binary label vector of length 2N (1 for related, 0 for not), and three 2N x 128 x 128 x 3 matrices of batches (one for father, one for mother and one for child)
# returns dad_Batch, mom_Batch, child_batch, labels
def shuffle_to_train(batch_fathers, batch_mothers, batch_pos_children, batch_neg_children):
    # randomly permute the fathers with their positive children, and the same fathers with their negative children. This is to be shuffled by the use of the np.random.permutation, by using the same seed.
    p = np.random.permutation(batch_fathers.shape[0] * 2) 
    bf = np.tile(batch_fathers, (2,1,1,1))
    bf = bf[p]
    bm = np.tile(batch_mothers, (2,1,1,1))
    bm = bm[p]
    bc = np.vstack((batch_pos_children, batch_neg_children))
    bc = bc[p]

    labels = np.concatenate((np.ones(batch_fathers.shape[0]), np.zeros(batch_fathers.shape[0])))[p]
    return bf, bm, bc, labels

