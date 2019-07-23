#!/usr/bin/env python

from model import Missing_Child_Model
import tensorflow as tf
import data_loader

EPOCHS = 5000
BATCHES_PER_EPOCH = 10
BATCH_SIZE = 50 # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 

tboard_logdir="./tboard_logs"

if __name__=='__main__':

    mcm = Missing_Child_Model()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    data_loader = data_reader()
    data_loader.set_traintest_split(0.7)

    summary_writer = tf.contrib.summary.create_file_writer(tboard_logdir, flush_millis=4000)
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for i in range(EPOCHS):
            avg_triplet_loss = 0
            for j in range(BATCHES_PER_EPOCH):
              	bf, bm, m_array, bcp, bcn = data_loader.get_next_train_batch(BATCH_SIZE)

                batch_loss = mcm.train_one_step(optimizer, \
                    bf, \
                    bm, \
                    m_array, \
                    bcp, \
                    bcn \
                )
                print("epoch {}, batch # {}, loss {}".format(i, j, batch_loss))
                avg_triplet_loss += batch_loss
            avg_triplet_loss /= (BATCHES_PER_EPOCH * 1.0)


            bf, bm, m_array, bc = data_loader.get_next_test_batch(BATCH_SIZE)
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

            # we include the step here because we don't do global steps. Is this the right way to do it actually?
            tf.contrib.summary.scalar('top_5_acc', top_5_acc, step=i)
            tf.contrib.summary.scalar('top_2_acc', top_2_acc, step=i)
            tf.contrib.summary.scalar('top_1_acc', top_1_acc, step=i)
            tf.contrib.summary.scalar('avg_triplet_loss', avg_triplet_loss, step=i)
