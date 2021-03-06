#!/usr/bin/env python

from model import Missing_Child_Model
import tensorflow as tf
import data_loader as data
from laedr.modeleag import Saver
import os

EPOCHS = 500
BATCHES_PER_EPOCH = 10
BATCH_SIZE = 50  # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 

TF_DEVICE=os.getenv("TF_DEVICE") if os.getenv("TF_DEVICE") is not None else "GPU:0"
learning_rate = float(os.getenv("LEARNING_RATE")) if os.getenv("LEARNING_RATE") is not None else 0.0001
print("using tf device {}, learning rate {}".format(TF_DEVICE, learning_rate))

tboard_logdir="./tboard_logs"

if __name__=='__main__':


    mcm = Missing_Child_Model()

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    data_loader = data.Data_reader()
    data_loader.set_traintest_split(0.75)

    summary_writer = tf.contrib.summary.create_file_writer(tboard_logdir, flush_millis=4000)

    model_saver = Saver(mcm, optimizer)
    # TODO: restore model if present.
    if os.path.isdir("./model_checkpoints"):
        print("restoring from ./model_checkpoints...")
        model_saver.restore("./model_checkpoints")
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        with tf.device(TF_DEVICE):
            for i in range(EPOCHS):
                avg_triplet_loss, avg_train_top10, avg_train_top5, avg_train_top2, avg_train_top1 = [0,0,0,0,0]
                for j in range(BATCHES_PER_EPOCH):
                    bf, bm, m_array, bcp, bcn = data_loader.get_next_train_batch(BATCH_SIZE)
                    batch_triplet_loss = mcm.train_one_step(optimizer, bf, bm, m_array, bcp, bcn)
                    top_10_train_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bcp, top_n = 10, cache_id="ep-{}-{}".format(i,j))
                    top_5_train_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bcp, top_n = 5, cache_id="ep-{}-{}".format(i,j))
                    top_2_train_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bcp, top_n = 2, cache_id="ep-{}-{}".format(i,j))
                    top_1_train_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bcp, top_n = 1, cache_id="ep-{}-{}".format(i,j))
                    print("epoch {}, batch # {}, triplet loss {}, top 10 train {}, top 5 train {}, top 2 train {}, top 1 train {}".format(i, j, batch_triplet_loss, top_10_train_acc, top_5_train_acc, top_2_train_acc, top_1_train_acc))
                    avg_triplet_loss += batch_triplet_loss 
                    avg_train_top10 += top_10_train_acc
                    avg_train_top5 += top_5_train_acc
                    avg_train_top2 += top_2_train_acc
                    avg_train_top1 += top_1_train_acc

                avg_triplet_loss /= (BATCHES_PER_EPOCH * 1.0)
                avg_train_top10 /= (BATCHES_PER_EPOCH * 1.0)
                avg_train_top5 /= (BATCHES_PER_EPOCH * 1.0)
                avg_train_top2 /= (BATCHES_PER_EPOCH * 1.0)
                avg_train_top1 /= (BATCHES_PER_EPOCH * 1.0)

                bf, bm, m_array, bc = data_loader.get_next_test_batch(BATCH_SIZE)
                # top 10
                top_10_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 10, cache_id="ep-{}".format(i))

                # top 5
                top_5_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 5, cache_id="ep-{}".format(i))

                # top 2
                top_2_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 2, cache_id="ep-{}".format(i))

                # top 1
                top_1_acc = mcm.evaluate_accuracy(BATCH_SIZE, bf, bm, m_array, bc, top_n = 1, cache_id="ep-{}".format(i))
                print("TOP-10 avg train ACC: {}, TOP-5 avg train ACC: {}, TOP-2 avg train ACC: {}, TOP-1 avg train ACC: {}".format(avg_train_top10, avg_train_top5, avg_train_top2, avg_train_top1))

                print("TOP-10 ACC: {}, TOP-5 ACC: {}, TOP-2 ACC: {}, TOP-1 ACC: {}".format(top_10_acc, top_5_acc, top_2_acc, top_1_acc))

                # we include the step here because we don't do global steps. Is this the right way to do it actually?
                tf.contrib.summary.scalar('top_10_train_acc', avg_train_top10, step=i)
                tf.contrib.summary.scalar('top_5_train_acc', avg_train_top5, step=i)
                tf.contrib.summary.scalar('top_2_train_acc', avg_train_top2, step=i)
                tf.contrib.summary.scalar('top_1_train_acc', avg_train_top1, step=i)

                tf.contrib.summary.scalar('top_10_test_acc', top_10_acc, step=i)
                tf.contrib.summary.scalar('top_5_test_acc', top_5_acc, step=i)
                tf.contrib.summary.scalar('top_2_test_acc', top_2_acc, step=i)
                tf.contrib.summary.scalar('top_1_test_acc', top_1_acc, step=i)

                tf.contrib.summary.scalar('avg_triplet_loss', avg_triplet_loss, step=i)

                # save the model every 2 epochs
                if i > 0 and i % 2 == 0:
                    model_saver.save("./model_checkpoints/ckpt-{}".format(i))

