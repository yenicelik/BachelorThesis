from h5py import File
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import numpy as np
# import matplotlib.pyplot as plt
# import os

from code.tripathy.src import config

TRAIN_TEST_SPLIT = (0.7, 0.3)


def import_test_train():
    f_eval = File(config['swissfel_datapath'], 'r')

    # tables = [k for k in f_eval.keys()]
    # print(f"recorded runs: {tables}")
    #
    # # tables = ['2'] # comment out if you want to use all tables
    #
    # print(f"using tables {tables}")
    #
    # # prepare data
    # data = None
    #
    # for t in tables:
    #     dset = f_eval[t]
    #     if data is None:
    #         data = dset[...].copy()
    #     else:
    #         data = np.concatenate((data, dset[...]))
    #
    # num_rows = len(data)
    # print(f"Found {num_rows} evaluations.")


if __name__ == "__main__":
    import_test_train()
