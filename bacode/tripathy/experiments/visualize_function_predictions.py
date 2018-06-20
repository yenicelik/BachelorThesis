"""
    Here, we create the vanilla visualizations for all functions.
"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# TODO: the following does not work for anything that's not integer!
from bacode.tripathy.experiments.utils import do_plotting, do_plotting_real_vs_gaussian
from bacode.tripathy.experiments.predictions import PredictRembo, PredictStiefelSimple, train_and_predict_all_models
from febo.environment.benchmarks.functions import ParabolaEmbedded2D, CamelbackEmbedded5D, \
    DecreasingSinusoidalEmbedded5D, RosenbrockEmbedded10D

# Generate the intervals at which we're visualizing each individual function (between the intervals of -1 and 1
points_per_axis = 100.


###############################
#           2D to 1D          #
###############################
def generate_train_test_data_2d_to_1d(fun, train_sampels, test_sampels):
    dim = fun.domain.d
    dcenter = fun.domain.u + fun.domain.l / 2.
    drange = fun.domain.u - fun.domain.l

    X_train = (np.random.rand(train_sampels, dim) * drange) + dcenter
    Y_train = fun.f(X_train.T).reshape(-1, 1)

    X_test = (np.random.rand(test_sampels, dim) * drange) + dcenter
    Y_test = fun.f(X_test.T).reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test

def visualize_2d_to_1d():
    function_instance = ParabolaEmbedded2D()

    lower = function_instance.domain.l
    upper = function_instance.domain.u

    print("Upper and lower are: ", lower, upper)

    X_train, Y_train, X_test, Y_test = generate_train_test_data_2d_to_1d(function_instance, train_sampels=100, test_sampels=2000)
    Y_real = function_instance.f(X_test.T)

    # Get the predicted datasets
    rembo_Yhat, tripathy_Yhat, boring_yhat = train_and_predict_all_models(X_train, Y_train, X_test, Y_test,
                                                                          function_instance.domain)

    # do_plotting_real_vs_gaussian("embedded_parabola_2d_to_1d_rembo", X_test, Y_real, rembo_Yhat)
    do_plotting_real_vs_gaussian("embedded_parabola_2d_to_1d_tripathy", X_test, Y_real, tripathy_Yhat)
    # do_plotting_real_vs_gaussian("embedded_parabola_2d_to_1d_boring", X_test, Y_real, boring_yhat)

###############################
#       5D to 2D VANILLA      #
###############################
def visualize_5d_to_2d_plain():
    function_instance = CamelbackEmbedded5D()

    lower = np.asarray([-2., -1.])
    upper = np.asarray([2., 1.])
    drange = upper - lower
    dcenter = (upper + lower) / 2.

    # lower = function_instance.domain.l
    # upper = function_instance.domain.u
    step_size = (upper - lower) / (points_per_axis)

    print("Upper and lower are: ", lower, upper)

    # Difference between meshgrid and random generation causes the weird function!
    Xs: np.ndarray = np.meshgrid(
                np.arange(lower[0], upper[0], step_size[0]),
                np.arange(lower[1], upper[1], step_size[1]),
            )
    X_test = np.vstack([X_ele.flatten() for X_ele in Xs]).T

    # X_test = (np.random.rand(100**2, 2) * drange) + dcenter
    X_train = (np.random.rand(100, 2) * drange) + dcenter

    print(X_test)
    X_test = np.dot(X_test, function_instance.W)
    X_train = np.dot(X_train, function_instance.W)
    print(X_test)

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X_test.shape)
    Y_train = function_instance.f(X_train.T).reshape(-1, 1)
    Y_test = function_instance.f(X_test.T).reshape(-1, 1)  # TODO: check if the input dimension is appropriate
    print("Y shape is: ", Y_test.shape)

    rembo_Yhat, tripathy_Yhat, boring_yhat = train_and_predict_all_models(X_train, Y_train, X_test, Y_test,
                                                                          function_instance.domain)
    X_vis = np.dot(X_test, function_instance.W.T)
    print(X_vis.shape)
    # do_plotting_real_vs_gaussian("embedded_sinusoidal_small_perturbations_5d_to_2d_rembo", X_vis, Y_test, rembo_Yhat)
    do_plotting_real_vs_gaussian("embedded_sinusoidal_small_perturbations_5d_to_2d_tripathy", X_vis, Y_test, tripathy_Yhat)
    # do_plotting_real_vs_gaussian("embedded_sinusoidal_small_perturbations_5d_to_2d_boring", X_vis, Y_test, Y_test)

###############################
# 5D to 2D SMALL PERTURBATION #
###############################
def visualize_5d_to_2d_small_perturbation():
    function_instance = DecreasingSinusoidalEmbedded5D()

    lower = np.asarray([-5., -5.])
    upper = np.asarray([10., 10.])
    drange = upper - lower
    dcenter = (upper + lower) / 2.

    # lower = function_instance.domain.l
    # upper = function_instance.domain.u
    step_size = (upper - lower) / (points_per_axis)

    print("Upper and lower are: ", lower, upper)

    X_test = (np.random.rand(100**2, 2) * drange) + dcenter
    X_train = (np.random.rand(100, 2) * drange) + dcenter

    print(X_test)
    X_test = np.dot(X_test, function_instance.W)
    X_train = np.dot(X_train, function_instance.W)
    print(X_test)

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X_test.shape)
    Y_train = function_instance.f(X_train.T).reshape(-1, 1)
    Y_test = function_instance.f(X_test.T).reshape(-1, 1)  # TODO: check if the input dimension is appropriate
    print("Y shape is: ", Y_test.shape)

    rembo_Yhat, tripathy_Yhat, boring_yhat = train_and_predict_all_models(X_train, Y_train, X_test, Y_test,
                                                                          function_instance.domain)
    X_vis = np.dot(X_test, function_instance.W.T)
    print(X_vis.shape)
    # do_plotting("embedded_sinusoidal_small_perturbations_5d_to_2d", X_vis, Y)
    do_plotting_real_vs_gaussian("embedded_sinusoidal_small_perturbations_5d_to_2d", X_vis, Y_test, tripathy_Yhat)

###############################
#          10D to 5D          #
###############################
def visualize_10d_to_5d():
    function_instance = RosenbrockEmbedded10D()

    lower = np.asarray([-5., -5., -5., -5., -5.])
    upper = np.asarray([10., 10., 10., 10., 10.])
    drange = upper - lower
    dcenter = (upper + lower) / 2.

    step_size = (upper - lower) / (points_per_axis / 14)

    print("Upper and lower are: ", lower, upper)

    X_test = (np.random.rand(100**2, 5) * drange) + dcenter
    X_train = (np.random.rand(100, 5) * drange) + dcenter

    print(X_test)
    X_test = np.dot(X_test, function_instance.W)
    X_train = np.dot(X_train, function_instance.W)
    print(X_test)

    # Project the points to the embeddings to apply a function evaluation
    print("X shape is: ", X_test.shape)
    Y_train = function_instance.f(X_train.T).reshape(-1, 1)
    Y_test = function_instance.f(X_test.T).reshape(-1, 1)
    print("Y shape is: ", Y_test.shape)

    rembo_Yhat, tripathy_Yhat, boring_yhat = train_and_predict_all_models(X_train, Y_train, X_test, Y_test,
                                                                          function_instance.domain)

    # Apply PCA on X!
    standardizedData = StandardScaler().fit_transform(X_test)
    pca = PCA(n_components=2)
    X_vis= pca.fit_transform(X=standardizedData)

    # do_plotting("embedded_rosenbrock_10d_to_5d", principalComponents, Y)
    do_plotting_real_vs_gaussian("embedded_sinusoidal_small_perturbations_5d_to_2d", X_vis, Y_test, tripathy_Yhat)

def main():
    print("Starting to visualize all functions")
    # print("Visualizing 2d to 1d plain")
    # visualize_2d_to_1d()
    # print("Visualizing 5d to 2d plain")
    # visualize_5d_to_2d_plain()
    # print("Visualizing 5d to 2d with small perturbations")
    # visualize_5d_to_2d_small_perturbation()
    print("Visualizing 10d to 5d")
    visualize_10d_to_5d()
    print("Done!")


if __name__ == "__main__":
    print("Starting to visualize all functions")
    main()
