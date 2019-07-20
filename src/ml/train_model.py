#!/usr/bin/env python

from model import Missing_Child_Model
import tensorflow as tf
import mock_data_loader

EPOCHS = 5000
BATCHES_PER_EPOCH = 100
BATCH_SIZE = 30 # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 


if __name__=='__main__':
    # TODO: load the data model here.

    mcm = Missing_Child_Model()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    mock_data_loader.set_traintest_split(0.7)
    for i in range(EPOCHS):
        for j in range(BATCHES_PER_EPOCH):
            
            # TEMP: mock data loading
            bf, bm, m_array, bcp, bcn= mock_data_loader.load_training_batch(BATCH_SIZE)
            batch_loss = mcm.train_one_step(optimizer, \
                bf, \
                bm, \
                m_array, \
                bcp, \
                bcn \
            )
            print("epoch {}, batch # {}, loss {}".format(i, j, batch_loss))

        # TODO: load test batch here.
        #TODO: should optimize on caching the accuracy matrices.
        #IDEA: implement an 'id' variable to use or invalidate the cache.
        # the id could be like (in this case ) the 'epoch' number, etc.

        # top 5
        #mcm.evaluate_accuracy(BATCH_SIZE, ..., top_n = 5)

        # top 2
        #mcm.evaluate_accuracy(BATCH_SIZE, ..., top_n = 2)

        # top 1
        #mcm.evaluate_accuracy(BATCH_SIZE, ..., top_n = 1)

        # TODO: After BATCH_SIZE batches, visualise some stuff here using tensorboard?
