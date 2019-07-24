#!/usr/bin/env python

from model import Missing_Child_Model
import tensorflow as tf
import data_loader as data
from laedr.modeleag import Saver

EPOCHS = 500
BATCHES_PER_EPOCH = 10
BATCH_SIZE = 40  # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 
TF_DEVICE="GPU:0"

tboard_logdir="./tboard_logs"

if __name__=='__main__':


    mcm = Missing_Child_Model()

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
    data_loader = data.Data_reader()
    data_loader.set_traintest_split(0.7)

    summary_writer = tf.contrib.summary.create_file_writer(tboard_logdir, flush_millis=4000)

    model_saver = Saver(mcm, optimizer)
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        with tf.device(TF_DEVICE):
            for i in range(EPOCHS):
                avg_triplet_loss = 0
                for j in range(BATCHES_PER_EPOCH):
                    bf, bm, m_array, bcp, bcn = data_loader.get_next_train_batch(BATCH_SIZE)
                    batch_triplet_loss = mcm.train_one_step(optimizer, bf, bm, m_array, bcp, bcn)
                    print("epoch {}, batch # {}, triplet loss {}".format(i, j, batch_triplet_loss))
                    avg_triplet_loss += batch_loss
                avg_triplet_loss /= (BATCHES_PER_EPOCH * 1.0)

                bf, bm, m_array, bc = data_loader.get_next_test_batch(BATCH_SIZE)
                # top 5
                top_5_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 5, cache_id="ep-{}".format(i))

                # top 2
                top_2_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 2, cache_id="ep-{}".format(i))

                # top 1
                top_1_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 1, cache_id="ep-{}".format(i))
                print("TOP-5 ACC: {}, TOP-2 ACC: {}, TOP-1 ACC: {}".format(top_5_acc, top_2_acc, top_1_acc))

                # we include the step here because we don't do global steps. Is this the right way to do it actually?
                tf.contrib.summary.scalar('top_5_acc', top_5_acc, step=i)
                tf.contrib.summary.scalar('top_2_acc', top_2_acc, step=i)
                tf.contrib.summary.scalar('top_1_acc', top_1_acc, step=i)
                tf.contrib.summary.scalar('avg_triplet_loss', avg_triplet_loss, step=i)

                # save the model every 2 epochs
                if i > 0 and i % 2 == 0:
                    model_saver.save("./model_checkpoints/ckpt-{}".format(i))

