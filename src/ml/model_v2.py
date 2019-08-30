from __future__ import print_function
#"""
from laedr import modeleag as laedrM
from laedr import network as laedrNetwork
#"""
import sklearn
import tensorflow as tf
import numpy as np
import sys
import os

pretrainedLaedrModelPath = './laedr/model/'

#""" TEMPDIS
class LAEDR_AIM(laedrM.Model):
    def initialize(self):
        self.encoder = laedrNetwork.EncoderNet()
        optim_default = tf.compat.v1.train.AdamOptimizer(0.0001)
        saver = laedrM.Saver(self, optim_default)
        saver.restore(pretrainedLaedrModelPath)
#"""



class Missing_Child_Model_v2:
    def __init__(self):
        self.LAEDR_model = LAEDR_AIM()
        self.evaluation_metrics = {}
        self.perm = None # for the permutation constant.


    def forward_pass(self, batch_fathers, batch_mothers, batch_children):
        # batch_fathers, batch_mothers is N x 128 x 128 x 3 where N is the number of samples.
        # extract Age Invariant Features (AIFs) from both mom and dad


         
        """
        m = tf.keras.layers.Conv2D( filters=6, kernel_size=5,activation='relu')(batch_mothers)
        m = tf.keras.layers.Conv2D( filters=6,kernel_size=5, activation='relu')(m)
        m = tf.keras.layers.Conv2D( filters=6,kernel_size=5, activation='relu')(m)
        m = tf.keras.layers.Conv2D( filters=1,kernel_size=5, activation='relu')(m)
        m = tf.keras.layers.Flatten()(m)

        mother_aif = m
        father_aif = m
        child_aif = m
        """
        #""" TEMPDIS
        mother_aif = self.LAEDR_model.encoder(batch_mothers)
        father_aif = self.LAEDR_model.encoder(batch_fathers)
        child_aif = self.LAEDR_model.encoder(batch_children)

        if os.getenv("ALLOW_GRADIENT_ENCODER") is None:
            # stop gradient so that we don't backpropagate till here.
            mother_aif = tf.stop_gradient(mother_aif)
            father_aif = tf.stop_gradient(father_aif)
            child_aif = tf.stop_gradient(child_aif)
        # here seems ok.
        #"""


        # concatenate both, so we can match input output.
        cm_concat = tf.concat([child_aif, mother_aif], 1)
        print("cm_concat np sum: {}".format(np.sum(cm_concat)))
        cf_concat = tf.concat([child_aif, father_aif], 1)
        print("cf_concat np sum: {}".format(np.sum(cf_concat)))
        # up until here ok.
        # CF concat stays constant accross multiple forward_passes. But cf_layer1 doesnt.


        # cm net:
        # PROBLEM: executing cm_layer1 dense twice like below produces diff results.
        cm_layer1 = tf.keras.layers.Dense(90,  activation=tf.nn.sigmoid, use_bias=False)(cm_concat)
        print("cm_layer1 np sum: {}".format(np.sum(cm_layer1)))
        cm_layer1 = tf.keras.layers.Dense(90,  activation=tf.nn.sigmoid, use_bias=False)(cm_concat)
        print("cm_layer1 np sum 2: {}".format(np.sum(cm_layer1)))





        cm_layer2 = tf.layers.dense(cm_layer1, 80,  activation=tf.nn.sigmoid, use_bias=True)
        cm_layer3 = tf.layers.dense(cm_layer2, 70,  activation=tf.nn.sigmoid, use_bias=True)
        cm_layer4 = tf.layers.dense(cm_layer3, 60,  activation=tf.nn.sigmoid, use_bias=True)

        # cf net:
        cf_layer1 = tf.layers.dense(cf_concat, 90,  activation=tf.nn.sigmoid, use_bias=False)
        print("cf_layer1 np sum: {}".format(np.sum(cf_layer1)))
        cf_layer1 = tf.layers.dense(cf_concat, 90,  activation=tf.nn.sigmoid, use_bias=False)
        print("cf_layer1 np sum 2: {}".format(np.sum(cf_layer1)))

        cf_layer2 = tf.layers.dense(cf_layer1, 80,  activation=tf.nn.sigmoid, use_bias=True)
        print("cf_layer2 np sum: {}".format(np.sum(cf_layer2)))
        cf_layer3 = tf.layers.dense(cf_layer2, 70,  activation=tf.nn.sigmoid, use_bias=True)
        cf_layer4 = tf.layers.dense(cf_layer3, 60,  activation=tf.nn.sigmoid, use_bias=True)


        # merge cm and cf nets.

        cf_cm_concat = tf.concat([cf_layer4, cm_layer4], 1)
        print("cf_cm_concat np sum: {}".format(np.sum(cf_cm_concat)))
        final_layer1= tf.layers.dense(cf_cm_concat, 100,  activation=tf.nn.sigmoid, use_bias=True)
        print("final_layer1 np sum: {}".format(np.sum(final_layer1)))
        final_layer2= tf.layers.dense(final_layer1, 75,  activation=tf.nn.sigmoid, use_bias=True)
        final_layer3= tf.layers.dense(final_layer2, 25,  activation=tf.nn.sigmoid, use_bias=True)
        final_layer4= tf.layers.dense(final_layer3, 1,  activation=tf.nn.sigmoid, use_bias=False)
        print("final_layer4: {}".format(final_layer4))
        
        model_output = final_layer4
        return model_output


    # note: compute crossentropy loss. Shuffle the numpy array 
    def compute_cxent_loss(self, batch_fathers, batch_mothers, batch_pos_children, batch_neg_children):
        assert batch_fathers.shape[0] == batch_mothers.shape[0] == batch_pos_children.shape[0] == batch_neg_children.shape[0]
        labels, preds = self.get_labels_preds(batch_fathers, batch_mothers, batch_pos_children, batch_neg_children)
        labels = np.reshape(labels, (labels.shape[0], 1))
        #return tf.compat.v1.losses.log_loss(labels, preds)
        return 0.23
        




    def get_labels_preds(self, batch_fathers, batch_mothers, batch_pos_children, batch_neg_children):
        if self.perm is None:
            self.perm = np.random.permutation(batch_fathers.shape[0] * 2)

        # randomly permute the fathers with their positive children, and the same fathers with their negative children. This is to be shuffled by the use of the np.random.permutation, by using the same seed.
        p = self.perm
        bf = np.tile(batch_fathers, (2,1,1,1))
        bf = bf[p]
        bm = np.tile(batch_mothers, (2,1,1,1))
        bm = bm[p]
        bc = np.vstack((batch_pos_children, batch_neg_children))
        bc = bc[p]

        labels = np.concatenate((np.ones(batch_fathers.shape[0]), np.zeros(batch_fathers.shape[0])))[p]
        # THE PROBLEM LIES IN THE VARIABILITY OF THE forward_pass FUNCTION! Why?
        preds = self.forward_pass( bf, bm, bc)
        return labels, preds



    def train_one_step(self, optimizer, batch_fathers, batch_mothers, batch_child_pos, batch_child_neg):
        #optimizer.minimize(lambda: self.compute_cxent_loss(batch_fathers, batch_mothers, batch_child_pos, batch_child_neg))
        #gradients, variables = zip(*optimizer.compute_gradients(lambda: self.compute_cxent_loss(batch_fathers, batch_mothers, batch_child_pos, batch_child_neg)))
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        #optimizer.apply_gradients(zip(gradients, variables))
        #print("bl 0 : {}".format(batch_loss))
        batch_loss = self.compute_cxent_loss(batch_fathers, batch_mothers, batch_child_pos, batch_child_neg)
        print("bl 1 : {}".format(batch_loss))
        batch_loss = self.compute_cxent_loss(batch_fathers, batch_mothers, batch_child_pos, batch_child_neg)
        print("bl 2 : {}".format(batch_loss))
        return batch_loss 



    # returns accuracy, precision, recall, f1, auroc
    def evaluate_metrics(self, batch_fathers, batch_mothers, batch_child_pos, batch_child_neg):
        labels, preds = self.get_labels_preds(batch_fathers, batch_mothers, batch_child_pos, batch_child_neg)
        rounded_preds = np.round(preds)
        accuracy = sklearn.metrics.accuracy_score(labels, rounded_preds)
        precision = sklearn.metrics.precision_score(labels, rounded_preds)
        recall = sklearn.metrics.recall_score(labels, rounded_preds)
        f1 = sklearn.metrics.f1_score(labels, rounded_preds)
        auroc = sklearn.metrics.roc_auc_score(labels, preds)

        return  accuracy, precision, recall, f1, auroc
