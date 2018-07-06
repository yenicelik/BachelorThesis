
import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bacode.tripathy.src.bilionis_refactor.config import config

def do_plotting(title, X, Y):
    assert X.shape[1] == 2, ("X is not plottable, as it is not 2-dimensional!", X.shape,)
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
    ax.scatter(X[:, 0], X[:, 1], Y, s=1)

    fig.savefig(config['visualize_vanilla_vs_gp_path'] + title)
    # plt.show()
    # plt.close(fig)
    plt.clf()


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
    ax.scatter(X[:, 0], X[:, 1], Y_gp.squeeze(), cmap=plt.cm.jet, s=5)

    fig.savefig(config['visualize_vanilla_vs_gp_path'] + title)
    # plt.show()
    # plt.close(fig)
    plt.clf()
