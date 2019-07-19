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
        self.mother_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        self.father_input = tf.placeholder(np.float32, (None, 128, 128, 3))

        
        self.positive_child_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        self.negative_child_input = tf.placeholder(np.float32, (None, 128, 128, 3))
        self.child_input = tf.placeholder(np.float32, (None,128,128,3))


        # value between 0 to 1. Father Likedness = 1 - mother_likedness
        mother_likedness = tf.placeholder(n.p.float32, (None,))

        # extract Age Invariant Features (AIFs) from both mom and dad
        mother_aif = self.LAEDR_model.encoder(self.mother_input)
        father_aif = self.LAEDR_model.encoder(self.mother_input)
        positive_child_aif = self.LAEDR_model.encoder(self.positive_child_input)
        negative_child_aif = self.LAEDR_model.encoder(self.negative_child_input)
        self.child_aif = self.LAEDR_model.encoder(child_input)

        #TODO: attention here

        # concatenate both, so we can match input output.
        mf_aif_concatted = tf.concat([mother_aif, father_aif], 0)

        # this is the input-output NN. Matches mom and dad concated
        layer1 = tf.layers.dense(mf_aif_concatted, 40, activation=tf.nn.tanh, use_bias=True)
        layer2 = tf.layers.dense(layer1, 30, activation=tf.nn.tanh, use_bias=True)
        layer3 = tf.layers.dense(layer2, 40, activation=tf.nn.tanh, use_bias=True)
        model_output = tf.layers.dense(layer3, laedrNetwork.Z_DIM, activation=tf.nn.tanh, use_bias=True)
        self.model_output = model_output

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

    def evaluate_accuracy(self, session, batch_size, batch_fathers, batch_mothers, batch_children, top_n = 1):
        if self.evaluation_metrics["top_{}_acc".format(top_n)] is None:
            # child_aif is N x D, model_output is N xD as well. 
            # Need N by N matrix. Every model output needs a distance with a real child. Then compute top n.

            model_output_squared_norms = tf.reduce_sum(tf.math.square(self.model_output), 1)
            child_aif_squared_norms = tf.reduce_sum(tf.math.square(self.child_aif), 1)
            squared_norms = model_output_squared_norms + child_aif_squared_norms

            squared_distance_matrix = squared_norms- 2 * tf.matmul(self.model_output, self.child_aif,  transpose_a=False, transpose_b=True)
            distance_matrix = tf.sqrt(squared_distance_matrix)
            candidates = - tf.nn.top_k(- distance_matrix, k=top_n, sorted=True)

            indices = tf.constant([i for i in range(batch_size)], dtype=np.float32, shape= [batch_size])
            diff = candidates- indices
            mask = tf.equal(diff, 0) # if any element is 0, that means the child is found.
            child_found_vec =tf.reduce_any(tf.boolean_mask(diff,mask), 1) # see if any child is found.

            self.evaluation_accuracy_score = tf.reduce_mean(child_found_vec)

        acc_score = session.run([self.evaluation_accuracy_score ] {self.father_input: batch_fathers,\
            self.mother_input: batch_mothers, self.child_input: batch_children \
        })



        

    





        
