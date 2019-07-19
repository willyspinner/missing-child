#!/usr/bin/env python

from laedr import modeleag as laedrM
from laedr import network as laedrNetwork
import tensorflow as tf

pretrainedLaedrModelPath = './laedr/model/'

class LAEDR_AIM(laedrM.Model):
	def initialize(self):
		self.encoder = laedrNetwork.EncoderNet()
		self.decoder = laedrNetwork.DecoderNet()
		self.dis_z = laedrNetwork.DiscriminatorZ()
		self.age_classifier = laedrNetwork.AgeClassifier() 
		self.dis_img = laedrNetwork.DiscriminatorPatch()
        optim_default = tf.train.AdamOptimizer(0.0001)
        saver = laedrM.Saver(self, optim_default)
        saver.restore(pretrainedLaedrModelPath)


class Missing_Child_Model:
    def __init__(self):
        self.LAEDR_model = LAEDR_AIM()

    def initialize(self):
        # inputs
        mother_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        father_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        child_input = tf.placeholder(np.float32, (None, 128, 128, 3))

        # value between 0 to 1. Father Likedness = 1 - mother_likedness
        mother_likedness = tf.placeholder(n.p.float32, (None,))

        # extract age invariant features from both mom and dad
        mother_aif = self.LAEDR_model.encoder(mother_input)
        father_aif = self.LAEDR_model.encoder(mother_input)
        child_aif = self.LAEDR_model.encoder(child_input)

        #TODO: attention here

        # concatenate both, so we can match input output.
        mf_aif_concatted = tf.concat([mother_aif, father_aif], 0)

        # this is the input-output NN. Matches mom and dad concated
        layer1 = tf.layers.dense(mf_aif_concatted, 40, activation=tf.nn.tanh, use_bias=True)
        layer2 = tf.layers.dense(layer1, 30, activation=tf.nn.tanh, use_bias=True)
        layer3 = tf.layers.dense(layer2, 40, activation=tf.nn.tanh, use_bias=True)
        model_output = tf.layers.dense(layer3, laedrNetwork.Z_DIM, activation=tf.nn.tanh, use_bias=True)

        # compute the loss 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))

        step = tf.train.AdamOptimizer().minimize(loss)
    

# triplet loss as said by VGGFACE : Maximising vector distance for unrelated pairs, and minimising otherwise.
def triplet_loss():
    #TODO.
    # Maybe see: https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
    pass




        
