#!/usr/bin/env python

from model import Missing_Child_Model

from keras_utils 
EPOCHS = 5000
BATCHES_PER_EPOCH = 100
BATCH_SIZE = 1000 # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 


if __name__=='__main__':
    # TODO: load the data model here.

    mcm = Missing_Child_Model()

    sess = tf.compat.v1.Session()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

    for i in range(EPOCHS):
        for j in range(BATCHES_PER_EPOCH):
            #TODO: get the batch from data loader and train one step:k
            batch_fathers, batch_mothers, mother_likedness_array, batch_child_positives, batch_child_negatives = load_data()
            batch_loss = mcm.train_one_step(optimizer, \
                batch_fathers, \
                batch_mothers, \
                mother_likedness_array, \
                batch_child_positives, \
                batch_child_negatives, \
            )
        # TODO: load test batch here.
        #TODO: should optimize on caching the accuracy matrices.
        #IDEA: implement an 'id' variable to use or invalidate the cache.
        # the id could be like (in this case ) the 'epoch' number, etc.

        # top 5
        mcm.evaluate_accuracy(BATCH_SIZE, ..., top_n = 5)

        # top 2
        mcm.evaluate_accuracy(BATCH_SIZE, ..., top_n = 2)

        # top 1
        mcm.evaluate_accuracy(BATCH_SIZE, ..., top_n = 1)

        # TODO: After BATCH_SIZE batches, visualise some stuff here using tensorboard?
