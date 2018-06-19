"""
    Here, we create the vanilla visualizations for all functions.
"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# TODO: the following does not work for anything that's not integer!
from bacode.tripathy.experiments.utils import do_plotting, do_plotting_real_vs_gaussian, generate_train_test_data
from bacode.tripathy.experiments.predictions import PredictRembo, PredictStiefelSimple, train_and_predict_all_models
from febo.environment.benchmarks.functions import ParabolaEmbedded2D, CamelbackEmbedded5D, DecreasingSinusoidalEmbedded5D, RosenbrockEmbedded10D

# Generate the intervals at which we're visualizing each individual function (between the intervals of -1 and 1
points_per_axis = 100.

###########################
# VISUALIZATION FUNCTIONS #
###########################
def visualize_2d_to_1d():
    function_instance = ParabolaEmbedded2D()

    lower = function_instance.domain.l
    upper = function_instance.domain.u
    step_size = (upper - lower ) / points_per_axis

    print("Upper and lower are: ", lower, upper)

    # For each dimension, we create a range!
    X1, X2 = np.meshgrid(
                np.arange(lower[0], upper[0], step_size[0]),
                np.arange(lower[1], upper[1], step_size[1])
            )
    X = np.vstack((X1.flatten(), X2.flatten())).T

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X.shape)
    Y = function_instance.f(X.T) # TODO: check if the input dimension is appropriate

    X_train, Y_train, X_test, Y_test = generate_train_test_data(function_instance.f, lower, upper, train_sampels=100, test_sampels=2000)
    Y_real = function_instance.f(X_test.T)

    # Get the predicted datasets
    rembo_Yhat, tripathy_Yhat, boring_yhat = train_and_predict_all_models(X_train, Y_train, X_test, Y_test, function_instance.domain)

    # print("Predicted Y is: ", rembo_Yhat.shape, tripathy_Yhat.shape, boring_yhat.shape)

    # do_plotting_real_vs_gaussian("embedded_parabola_2d_to_1d_rembo", X_test, Y_real, rembo_Yhat)
    # do_plotting_real_vs_gaussian("embedded_parabola_2d_to_1d_tripathy", X_test, Y_real, tripathy_Yhat)
    do_plotting_real_vs_gaussian("embedded_parabola_2d_to_1d_boring", X_test, Y_real, boring_yhat)

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
                np.arange(lower[0], upper[0], step_size[0]),
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

def visualize_5d_to_2d_small_perturbation():
    function_instance = DecreasingSinusoidalEmbedded5D()

    lower = np.asarray([-5., -5.])
    upper = np.asarray([10., 10.])

    # lower = function_instance.domain.l
    # upper = function_instance.domain.u
    step_size = (upper - lower ) / (points_per_axis)

    print("Upper and lower are: ", lower, upper)

    # For each dimension, we create a range!
    Xs: np.ndarray = np.meshgrid(
                np.arange(lower[0], upper[0], step_size[0]),
                np.arange(lower[1], upper[1], step_size[1]),
            )
    X = np.vstack([X_ele.flatten() for X_ele in Xs]).T
    print(X)
    X = np.dot(X, function_instance.W)
    print(X)

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X.shape)
    Y = function_instance.f(X.T) # TODO: check if the input dimension is appropriate
    print("Y shape is: ", Y.shape)

    X_vis = np.dot(X, function_instance.W.T)
    print(X_vis.shape)
    do_plotting("embedded_sinusoidal_small_perturbations_5d_to_2d", X_vis, Y)

def visualize_10d_to_5d():
    function_instance = RosenbrockEmbedded10D()

    lower = np.asarray([-5., -5., -5., -5., -5.])
    upper = np.asarray([10., 10., 10., 10., 10.])

    step_size = (upper - lower ) / (points_per_axis / 11)

    print("Upper and lower are: ", lower, upper)

    # For each dimension, we create a range!
    Xs: np.ndarray = np.meshgrid(
                np.arange(lower[0], upper[0], step_size[0]),
                np.arange(lower[1], upper[1], step_size[1]),
                np.arange(lower[2], upper[2], step_size[2]),
                np.arange(lower[3], upper[3], step_size[3]),
                np.arange(lower[4], upper[4], step_size[4]),
            )
    X = np.vstack([X_ele.flatten() for X_ele in Xs]).T
    print(X)
    X = np.dot(X, function_instance.W)
    print(X)

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X.shape)
    Y = function_instance.f(X.T) # TODO: check if the input dimension is appropriate
    print("Y shape is: ", Y.shape)

    X_vis = np.dot(X, function_instance.W.T)
    # Apply PCA on X!
    standardizedData = StandardScaler().fit_transform(X_vis)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X=standardizedData)

    do_plotting("embedded_rosenbrock_10d_to_5d", principalComponents, Y)


def main():
    print("Starting to visualize all functions")
    print("Visualizing 2d to 1d plain")
    visualize_2d_to_1d()
    # print("Visualizing 5d to 2d plain")
    # visualize_5d_to_2d_plain()
    # print("Visualizing 5d to 2d with small perturbations")
    # visualize_5d_to_2d_small_perturbation()
    # print("Visualizing 10d to 5d")
    # visualize_10d_to_5d()
    print("Done!")


if __name__ == "__main__":
    print("Starting to visualize all functions")
    main()
