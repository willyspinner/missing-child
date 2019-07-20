from laedr import modeleag as laedrM
from laedr import network as laedrNetwork
import tensorflow as tf
import numpy as np
import sys

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
        self.evaluation_metrics = {}
        self.step = None



    def forward_pass(self, batch_fathers, batch_mothers, mother_likedness_array):
        # batch_fathers, batch_mothers is N x 128 x 128 x 3 where N is the number of samples.
        # extract Age Invariant Features (AIFs) from both mom and dad
        mother_aif = self.LAEDR_model.encoder(batch_mothers)
        father_aif = self.LAEDR_model.encoder(batch_fathers)
        # stop gradient so that we don't backpropagate till here.
        tf.stop_gradient(mother_aif)
        tf.stop_gradient(father_aif)

        #TODO: implement attention here

        # concatenate both, so we can match input output.
        mf_aif_concatted = tf.concat([mother_aif, father_aif], 1)

        # this is the input-output NN. Matches mom and dad concated
        layer1 = tf.layers.dense(mf_aif_concatted, 40, activation=tf.nn.tanh, use_bias=True)
        layer2 = tf.layers.dense(layer1, 30, activation=tf.nn.tanh, use_bias=True)
        layer3 = tf.layers.dense(layer2, 40, activation=tf.nn.tanh, use_bias=True)
        model_output = tf.layers.dense(layer3, laedrNetwork.Z_DIM, activation=tf.nn.tanh, use_bias=True)
        return model_output


    def compute_rmse_loss(self, batch_fathers, batch_mothers, mother_likedness_array, batch_children):
        model_output = self.forward_pass(batch_fathers, batch_mothers, mother_likedness_array)
        return tf.sqrt(tf.reduce_mean(tf.square(model_output - batch_children)))


    def compute_triplet_loss(self, batch_fathers, batch_mothers, mother_likedness_array, batch_child_positives, batch_child_negatives):
        positive_child_aifs = self.LAEDR_model.encoder(batch_child_positives)
        negative_child_aifs = self.LAEDR_model.encoder(batch_child_negatives)
        model_output = self.forward_pass(batch_fathers, batch_mothers, mother_likedness_array)
        # triplet loss on the AIFs 
        # triplet loss as introduced by VGGFACE : Maximising vector distance for unrelated pairs, and minimising otherwise.
        d_pos = tf.reduce_sum(tf.square(model_output - positive_child_aifs), 1)
        d_neg = tf.reduce_sum(tf.square(model_output - negative_child_aifs), 1)
        triplet_loss_margin = tf.Variable(0.01, name="triplet_loss_margin")
        triplet_loss = tf.maximum(0., triplet_loss_margin + tf.reduce_sum(d_pos - d_neg))

        return triplet_loss


    # ANOTHER IDEA: classification cxent loss above some distance similarity threshold.
    #i.e. if a vector is less than 0.4 distance units, then classified positive. Else negative.



    # train one batch. Returns the batch loss.
    def train_one_step(self, optimizer, batch_fathers, batch_mothers, mother_likedness_array, batch_child_positives, batch_child_negatives):
    # TODO: also try RMSE loss of  the output and the child's AIF
        runner = optimizer.minimize(lambda: self.compute_triplet_loss(batch_fathers, batch_mothers, mother_likedness_array, batch_child_positives, batch_child_negatives))
        #TODO: for some reason, 'runner' here is None. Why? Aren't we supposed to run to update the gradients?
        #batch_loss = runner.run()
        batch_loss = self.compute_triplet_loss(batch_fathers, batch_mothers, mother_likedness_array, batch_child_positives, batch_child_negatives)
        return batch_loss





    #TODO: should optimize on caching the accuracy matrices.
    #IDEA: implement an 'id' variable to use or invalidate the cache.
    # the id could be like (in this case ) the 'epoch' number, etc.
    def evaluate_accuracy(self, batch_size, batch_fathers, batch_mothers, mother_likedness_array, batch_children, top_n = 1):
            child_aif = self.LAEDR_model.encoder(batch_children)
            tf.stop_gradient(child_aif)
            # child_aif is N x D, model_output is N xD as well. 
            # Need N by N matrix. Every model output needs a distance with a real child. Then compute top n.

            model_output = self.forward_pass(batch_fathers, batch_mothers, mother_likedness_array)

            model_output_squared_norms = tf.reduce_sum(tf.math.square(model_output), 1)
            child_aif_squared_norms = tf.reduce_sum(tf.math.square(child_aif), 1)
            squared_norms = model_output_squared_norms + child_aif_squared_norms

            squared_distance_matrix = squared_norms- 2 * tf.matmul(model_output, child_aif,  transpose_a=False, transpose_b=True)
            distance_matrix = tf.sqrt(squared_distance_matrix)
            _, candidates = tf.nn.top_k(tf.negative(distance_matrix), k=top_n, sorted=True)
            indices = tf.constant([i for i in range(batch_size)], dtype=np.int32, shape= [batch_size])
            indices = tf.reshape(indices, [30, 1])
            # i am paranoid so I put the below. You don't need it because broadcasting i
            # is implicit.
            indices = tf.broadcast_to(indices, tf.shape(candidates))
            diff = candidates - indices

            diff = tf.equal(diff, 0) # if any element is 0, that means the child is found.
            child_found_vec =tf.reduce_any(diff, 1) # see if any child is found.
            child_found_vec = tf.cast(child_found_vec, np.float32)
            acc_score = tf.reduce_mean(child_found_vec)

            return acc_score

        

    





        
