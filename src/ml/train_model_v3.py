#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.keras.backend import set_session
from model_v3 import create_model_v3
import data_loader as data

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tf.compat.v1.enable_eager_execution(config=config)
set_session(tf.get_default_session())

EPOCHS = 1200
BATCHES_PER_EPOCH = 20
BATCH_SIZE = 100  # this is the amount of samples for each M, F, CP, CN to be considered at each step. Note that because each MF pair has a positive and negative child, the training set batch size is actually 2 * BATCH_SIZE.

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
    with tf.device(TF_DEVICE):
        # train per batch
        mcm_model.fit_generator(data.train_generator(data_loader, BATCH_SIZE), epochs=EPOCHS, steps_per_epoch=BATCHES_PER_EPOCH)




