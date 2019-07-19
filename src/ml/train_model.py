#!/usr/bin/env python

from model import Missing_Child_Model

EPOCHS = 50000
BATCH_SIZE = 1000 # this is the amount of samples, where each consists of 1 M, 1 F, 1 C_P  1C_N . 


if __name__=='__main__':
    # TODO: load the data model here.

    mcm = Missing_Child_Model()

    for i in range(EPOCHS):
        #TODO:
        batch_loss = mcm.train_one_step()

        if i % 100 == 0:
            # TODO: visualise some stuff here using tensorboard?
        mcm.evaluate_accuracy()
