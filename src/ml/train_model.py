#!/usr/bin/env python

from model import Missing_Child_Model

from keras_utils 
EPOCHS = 5000
BATCHES_PER_EPOCH = 100
BATCH_SIZE = 1000 # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 


if __name__=='__main__':
    # TODO: load the data model here.

    mcm = Missing_Child_Model()
    mcm.initialize()

    sess = tf.compat.v1.Session()

    for i in range(EPOCHS):
        for j in range(BATCHES_PER_EPOCH):
            #TODO: get the batch from data loader and train one step:k
            batch_loss = mcm.train_one_step(sess, )

        # TODO: After BATCH_SIZE batches, visualise some stuff here using tensorboard?
        mcm.evaluate_accuracy()
