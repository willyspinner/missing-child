#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
from model_v2 import Missing_Child_Model_v2
import data_loader as data
#from laedr.modeleag import Saver


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tf.compat.v1.enable_eager_execution(config=config)

EPOCHS = 1200
BATCHES_PER_EPOCH = 25
BATCH_SIZE = 1  # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 

TF_DEVICE=os.getenv("TF_DEVICE") if os.getenv("TF_DEVICE") is not None else "GPU:0"
learning_rate = float(os.getenv("LEARNING_RATE")) if os.getenv("LEARNING_RATE") is not None else 0.0001

print("using tf device {}, learning rate {}".format(TF_DEVICE, learning_rate))

tboard_logdir="./tboard_logs"

if __name__=='__main__':

    mcm = Missing_Child_Model_v2()

    #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    data_loader = data.Data_reader()
    data_loader.set_traintest_split(0.75)

    summary_writer = tf.contrib.summary.create_file_writer(tboard_logdir, flush_millis=4000)

    #model_saver = Saver(mcm, optimizer)
    #if os.path.isdir("./model_checkpoints_v2"):
        #print("restoring from ./model_checkpoints_v2...")
        #model_saver.restore("./model_checkpoints_v2")
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        with tf.device(TF_DEVICE):
            for i in range(EPOCHS):
                avg_train_cxent_loss = 0
                for j in range(BATCHES_PER_EPOCH):
                    bf, bm, _, bcp, bcn = data_loader.get_next_train_batch(BATCH_SIZE)
                    optimizer = 2
                    batch_cxent_loss = mcm.train_one_step(optimizer, bf, bm, bcp, bcn)
                    print("epoch {} batch {}, train cxent loss: {}".format(i, j, batch_cxent_loss))
                    avg_train_cxent_loss += batch_cxent_loss

                avg_train_cxent_loss /= (BATCHES_PER_EPOCH * 1.0)
                print("epoch {}, avg train cxent loss: {}".format(i, avg_train_cxent_loss))
                bf, bm, _, bc = data_loader.get_next_test_batch(BATCH_SIZE)

                # evaluate.
                p_eval_c = np.random.permutation(bc.shape[0])
                acc, prec, recall, f1, auroc = mcm.evaluate_metrics(bf, bm, bc, bc[p_eval_c])
                print("test acc: {}, test prec {}, test recall {}, test f1 {}, test auroc {}".format(acc, prec, recall, f1, auroc))

                # we include the step here because we don't do global steps. Is this the right way to do it actually?
                tf.contrib.summary.scalar('avg_cxent_training_loss', avg_train_cxent_loss, step=i)
                tf.contrib.summary.scalar('acc', acc, step=i)
                tf.contrib.summary.scalar('prec', prec, step=i)
                tf.contrib.summary.scalar('recall', recall, step=i)
                tf.contrib.summary.scalar('f1', f1, step=i)
                tf.contrib.summary.scalar('auroc', auroc, step=i)


                # save the model every 2 epochs
                if i > 0 and i % 2 == 0:
                    model_saver.save("./model_checkpoints_v2/ckpt-{}".format(i))

