"""
    Here, we create the vanilla visualizations for all functions.
"""
import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO: the following does not work for anything that's not integer!
from bacode.tripathy.src.bilionis_refactor.config import config
from febo.utils.utils import cartesian
from febo.environment.benchmarks.functions import ParabolaEmbedded2D, CamelbackEmbedded5D, DecreasingSinusoidalEmbedded5D

# Generate the intervals at which we're visualizing each individual function (between the intervals of -1 and 1
points_per_axis = 100.

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

def visualize_2d_to_1d():
    function_instance = ParabolaEmbedded2D()

    lower = function_instance.domain.l
    upper = function_instance.domain.u
    step_size = (upper - lower ) / points_per_axis

    print("Upper and lower are: ", lower, upper)

    # For each dimension, we create a range!
    X1, X2 = np.meshgrid(
                np.arange(lower[0], upper[1], step_size[0]),
                np.arange(lower[1], upper[1], step_size[1])
            )
    X = np.vstack((X1.flatten(), X2.flatten())).T

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X.shape)
    Y = function_instance.f(X.T) # TODO: check if the input dimension is appropriate

    do_plotting("embedded_parabola_2d_to_1d", X, Y)

def visualize_5d_to_2d_plain():
    function_instance = CamelbackEmbedded5D()

    lower = np.asarray([-2., -1.])
    upper = np.asarray([2., 1.])

    # lower = function_instance.domain.l
    # upper = function_instance.domain.u
    step_size = (upper - lower ) / (points_per_axis)

    print("Upper and lower are: ", lower, upper)

    # For each dimension, we create a range!
    Xs: np.ndarray = np.meshgrid(
                np.arange(lower[0], upper[1], step_size[0]),
                np.arange(lower[1], upper[1], step_size[1]),
            )
    X = np.vstack([X_ele.flatten() for X_ele in Xs]).T
    X = np.dot(X, function_instance.W)
    print(X)

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X.shape)
    Y = function_instance.f(X.T) # TODO: check if the input dimension is appropriate
    print("Y shape is: ", Y.shape)

    X_vis = np.dot(X, function_instance.W.T)
    print(X_vis.shape)
    do_plotting("embedded_camelback_5d_to_2d", X_vis, Y)


def main():
    print("Starting to visualize all functions")
    #print("Visualizing 2d to 1d plain")
    #visualize_2d_to_1d()
    print("Visualizing 5d to 2d plain")
    visualize_5d_to_2d_plain()
    print("Done!")


if __name__ == "__main__":
    print("Starting to visualize all functions")
    main()
