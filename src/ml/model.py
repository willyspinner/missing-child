from laedr import modeleag as laedrM
from laedr import network as laedrNetwork
import tensorflow as tf
import numpy as np

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
        # inputs. Note: mother and father pair is the anchor.
        mother_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        father_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        
        positive_child_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        negative_child_input = tf.placeholder(np.float32, (None, 128, 128, 3))

        self.mother_input = mother_input
        self.father_input = father_input 
        
        self.positive_child_input = positive_child_input
        self.negative_child_input = negative_child_input

        # value between 0 to 1. Father Likedness = 1 - mother_likedness
        mother_likedness = tf.placeholder(n.p.float32, (None,))

        # extract Age Invariant Features (AIFs) from both mom and dad
        mother_aif = self.LAEDR_model.encoder(mother_input)
        father_aif = self.LAEDR_model.encoder(mother_input)
        positive_child_aif = self.LAEDR_model.encoder(positive_child_input)
        negative_child_aif = self.LAEDR_model.encoder(negative_child_input)

        #TODO: attention here

        # concatenate both, so we can match input output.
        mf_aif_concatted = tf.concat([mother_aif, father_aif], 0)

        # this is the input-output NN. Matches mom and dad concated
        layer1 = tf.layers.dense(mf_aif_concatted, 40, activation=tf.nn.tanh, use_bias=True)
        layer2 = tf.layers.dense(layer1, 30, activation=tf.nn.tanh, use_bias=True)
        layer3 = tf.layers.dense(layer2, 40, activation=tf.nn.tanh, use_bias=True)
        model_output = tf.layers.dense(layer3, laedrNetwork.Z_DIM, activation=tf.nn.tanh, use_bias=True)

        # triplet loss on the AIFs 
        # triplet loss as introduced by VGGFACE : Maximising vector distance for unrelated pairs, and minimising otherwise.
        d_pos = tf.reduce_sum(tf.square(model_output- positive_child_aif), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_child_aif), 1)
        triplet_loss_margin = tf.Variable(0.01, name="triplet_loss_margin")
        triplet_loss = tf.maximum(0., triplet_loss_margin + d_pos - d_neg)

        self.loss = triplet_loss
        self.step = None

    # train one batch. Returns the batch loss.
    def train_one_step(self, session, batch_fathers, batch_mothers, batch_child_positives, batch_child_negatives):
        if self.step is None:
            self.step = tf.train.AdamOptimizer().minimize(self. loss)

        _, batch_loss = session.run([self.step, self.loss], {self.father_input: batch_fathers, \
            self.mother_input: batch_mothers, self.positive_child_input: batch_child_positives, \
            self.negative_child_input: batch_child_negatives \
        )
        return batch_loss

    def evaluate_accuracy(self):
        #TODO: some evaluation function to be called with test set. 
        #Plot in terms of threshold?
        pass


        

    





        
