
import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bacode.tripathy.src.bilionis_refactor.config import config

def do_plotting(title, X, Y):
    assert X.shape[1] == 2, ("X is not plottable, as it is not 2-dimensional!")
    # assert Y.shape[1] == 1, ("Y is not plottable, as it is not 1-dimensional!")
    assert isinstance(title, str), ("Title is not a string!")

    # Visualize the function
    if not os.path.exists(config['visualize_vanilla_path']):
        os.makedirs(config['visualize_vanilla_path'])

    #################################
    #   END TRAIN THE W_OPTIMIZER   #
    #################################

    fig = plt.figure()
    ax = Axes3D(fig)

    # First plot the real function
    ax.scatter(X[:, 0], X[:, 1], Y, s=1)

    fig.savefig(config['visualize_vanilla_path'] + title)
    plt.show()
    plt.close(fig)

def do_plotting_real_vs_gaussian(title, X, Y_real, Y_gp):

    assert X.shape[1] == 2, ("X is not plottable, as it is not 2-dimensional!")
    # assert Y.shape[1] == 1, ("Y is not plottable, as it is not 1-dimensional!")
    assert isinstance(title, str), ("Title is not a string!")

    # Visualize the function
    if not os.path.exists(config['visualize_vanilla_vs_gp_path']):
        os.makedirs(config['visualize_vanilla_vs_gp_path'])

    #################################
    #   END TRAIN THE W_OPTIMIZER   #
    #################################

    fig = plt.figure()
    ax = Axes3D(fig)

    # First plot the real function
    ax.scatter(X[:, 0], X[:, 1], Y_real.squeeze(), 'k.', alpha=.3, s=1)
    ax.scatter(X[:, 0], X[:, 1], Y_gp.squeeze(), cmap=plt.cm.jet)

    fig.savefig(config['visualize_vanilla_vs_gp_path'] + title)
    plt.show()
    plt.close(fig)

def generate_train_test_data(fun, train_sampels, test_sampels, lower_dim=2):
    dim = fun.domain.d
    dcenter = fun.domain.u + fun.domain.l / 2.
    drange = fun.domain.u - fun.domain.l

    if hasattr(fun, 'W') and dim > 2:
        X_train = np.random.rand(train_sampels, lower_dim)
        X_train = (np.dot(X_train, fun.W) * drange ) + dcenter
        Y_train = fun.f(X_train.T).reshape(-1, 1)

        X_test = np.random.rand(test_sampels, lower_dim)
        X_test = (np.dot(X_test, fun.W) * drange ) + dcenter
        Y_test = fun.f(X_test.T).reshape(-1, 1)
    else:
        X_train = (np.random.rand(train_sampels, dim) * drange) + dcenter
        Y_train = fun.f(X_train.T).reshape(-1, 1)

        X_test = (np.random.rand(test_sampels, dim) * drange) + dcenter
        Y_test = fun.f(X_test.T).reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test