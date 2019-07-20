import numpy as np
def set_traintest_split(split):
    print("MOCK_DATA_LOADER: set traintest_split: {}".format(split))

def load_train_batch(batch_size):
    # father, mother, mom_likedness, positive kids, negative kids.
    return \
        np.random.rand(batch_size, 128, 128, 3), \
        np.random.rand(batch_size, 128, 128, 3), \
        np.random.rand(batch_size),\
        np.random.rand(batch_size, 128, 128, 3), \
        np.random.rand(batch_size, 128, 128, 3)

def load_test_batch(batch_size):
    # father, mother, mom_likedness, children.
    return \
        np.random.rand(batch_size, 128, 128, 3), \
        np.random.rand(batch_size, 128, 128, 3), \
        np.random.rand(batch_size),\
        np.random.rand(batch_size, 128, 128, 3)
