#!/usr/bin/env python

from model import Missing_Child_Model
import tensorflow as tf
import mock_data_loader

EPOCHS = 5000
BATCHES_PER_EPOCH = 10
BATCH_SIZE = 50 # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 


if __name__=='__main__':
    # TODO: load the data model here.

    mcm = Missing_Child_Model()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    mock_data_loader.set_traintest_split(0.7)
    for i in range(EPOCHS):
        for j in range(BATCHES_PER_EPOCH):
            
            # TEMP: mock data loading
            bf, bm, m_array, bcp, bcn= mock_data_loader.load_train_batch(BATCH_SIZE)
            batch_loss = mcm.train_one_step(optimizer, \
                bf, \
                bm, \
                m_array, \
                bcp, \
                bcn \
            )
            print("epoch {}, batch # {}, loss {}".format(i, j, batch_loss))


        bf, bm, m_array, bc = mock_data_loader.load_test_batch(BATCH_SIZE)
        # top 5
        top_5_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 5, cache_id="ep-{}".format(i))

        # top 2
        top_2_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 2, cache_id="ep-{}".format(i))

        # top 1
        top_1_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 1, cache_id="ep-{}".format(i))
        print("TOP-5 ACC: {}, TOP-2 ACC: {}, TOP-1 ACC: {}".format( \
            top_5_acc, \
            top_2_acc, \
            top_1_acc  \
        ))

        # TODO: After BATCH_SIZE batches, visualise some stuff here using tensorboard?
