#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
from model_v3 import create_model_v3 
from tensorflow.python.keras.backend import set_session

import data_loader as data


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tf.compat.v1.enable_eager_execution(config=config)
set_session(tf.get_default_session())

EPOCHS = 1200
BATCHES_PER_EPOCH = 25
BATCH_SIZE = 1  # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 

TF_DEVICE=os.getenv("TF_DEVICE") if os.getenv("TF_DEVICE") is not None else "GPU:0"
learning_rate = float(os.getenv("LEARNING_RATE")) if os.getenv("LEARNING_RATE") is not None else 0.0001

print("using tf device {}, learning rate {}".format(TF_DEVICE, learning_rate))

tboard_logdir="./tboard_logs"

if __name__=='__main__':

    mcm_model = create_model_v3() 

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    mcm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print("model instantiated")

    data_loader = data.Data_reader()
    data_loader.set_traintest_split(0.75)


    #NOTE: i stripped off everything to get the basics right.
    # if things work, implement the following:
    # - checkpointing
    # - tf summary writer (for tensorboard)
    # - batch learning, etc.
    with tf.device(TF_DEVICE):
        bf, bm, _, bcp, bcn = data_loader.get_next_train_batch(BATCH_SIZE)

        print("size: bf", bf.shape)
        print("size: bm", bm.shape)
        print("size: bcp", bcp.shape)
        print("size: bcn", bcn.shape)
        #WORKS:! It's working.
        mcm_model.fit([bf, bm, bcp], [1],batch_size=1, epochs=10000) 




