from h5py import File
import h5py
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import numpy as np
# import matplotlib.pyplot as plt
# import os

from bacode.tripathy.src.config import config

TRAIN_TEST_SPLIT = (0.7, 0.3)


def import_X_Y():
    f_eval = h5py.File(config['swissfel_datapath'], 'r')
    print("Successfully opened file... ")

    tables = [k for k in f_eval.keys()]
    print(f"recorded runs: {tables}")

    # tables = ['2'] # comment out if you want to use all tables

    print(f"using tables {tables}")

    # prepare data
    data = None

    for t in tables:
        dset = f_eval[t]
        if data is None:
            data = dset[...].copy()
        else:
            data = np.concatenate((data, dset[...]))

    num_rows = len(data)
    print(f"Found {num_rows} evaluations.")

    Y = data['y'].reshape(-1, 1)
    Y_std = data['y_std']
    X = data['x']
    steps = np.arange(len(Y))

    # normalize data somehow, e.g.
    Y_normalized = (Y - np.mean(Y)) / np.std(Y)
    Y_std = np.std(Y)

    print("Importing swissfel dataset with shapes: ", (X.shape, Y_normalized.shape, Y_std.shape))

    return X, Y_normalized.squeeze(), Y_std


if __name__ == "__main__":
    print("Staring imports...")
    import_test_train()
