# The following object includes the function, and domain information for each tuple
from febo.environment.benchmarks import ParabolaEmbedded2D, DecreasingSinusoidalEmbedded5D, \
    CamelbackEmbedded5D, CamelbackEmbedded4D, \
    CamelbackEmbedded3D, CamelbackEmbedded2D

from bacode.tripathy.experiments.angle_optimization.utils import visualize_angle_given_W_array
from bacode.tripathy.experiments.visualize_with_predictions.utils import do_plotting_real_vs_gaussian, do_plotting
from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel

import numpy as np

from bacode.tripathy.src.bilionis_refactor.t_optimizer import run_two_step_optimization

from bacode.tripathy.src.rembo.rembo_algorithm import RemboAlgorithm
from bacode.tripathy.src.boring.boring_algorithm import BoringGP
from bacode.tripathy.src.tripathy__ import TripathyGP

import sys

FNC_TUPLES = [
    # ["Parabola-2D->1D", ParabolaEmbedded2D(), 1],
    # ["Camelback-2D->2D", CamelbackEmbedded2D(), 2],
    # ["Camelback-3D->2D", CamelbackEmbedded3D(), 2],
    # ["Camelback-4D->2D", CamelbackEmbedded4D(), 2],
    # ["Camelback-5D->2D", CamelbackEmbedded5D(), 2],
    ["Sinusoidal-5D->2D", DecreasingSinusoidalEmbedded5D(), 2]
]


# Initialize the different algorithms
def initialize_rembo(domain):
    algo = RemboAlgorithm()
    algo.initialize(**{
        'domain': domain
    })
    return algo


def initialize_tripathy(domain):
    algo = TripathyGP(domain, calculate_always=True)
    return algo


def initialize_boring(domain):
    algo = BoringGP(domain, always_calculate=True)
    return algo


# Helper algorithms to generate test and training set
def get_training_set(points, fnc):
    # Generate training set
    np.set_printoptions(suppress=True)
    X_train = np.random.rand(points, fnc.d) - 0.5 # TODO: do we need to multiply by 2 first, bcs rn it's [-0.5, 0.5]!
    print("Center is: ", fnc._domain.center)
    print("Range is: ", fnc._domain.range)
    X_train = (X_train * fnc._domain.range) + fnc._domain.center
    Y_train = fnc.f(X_train.T).reshape(-1, 1)
    print("Shape of X, Y: ", (X_train.shape, Y_train.shape))
    # print("X train is: ", X_train)
    print("Max and then Min are: ")
    print(np.max(X_train, axis=0))
    print(np.min(X_train, axis=0))
    return X_train, Y_train


def get_test_set(points, fnc):
    # Figure out what the domain is in the lower-dimensional subspace
    lower = fnc._domain.l
    upper = fnc._domain.u

    print("Shape of lower and higher are")
    print(lower.shape)

    if fnc._domain.d > 2:
        lower = np.dot(fnc.W, lower)
        upper = np.dot(fnc.W, upper)

    print("Shape of lower and higher are")
    print(lower.shape)

    assert lower.shape == (2,), ("lower does not have the right shape!")

    # Calculate steps (depending on range) - should be 100 points per dimension
    stepsize = (upper - lower) / float(points)

    # Create a meshgrid in the real coordinate system
    Xs = np.meshgrid(
        np.arange(lower[0], upper[0], stepsize[0]),
        np.arange(lower[1], upper[1], stepsize[1])
    )
    X_viz = np.vstack([X_ele.flatten() for X_ele in Xs]).T

    assert X_viz.shape[1] == 2, ("Vizualizable X is not 2D", X_viz.shape)

    if fnc._domain.d > 2:
        X_test = np.dot(X_viz, fnc.W)
    else:
        X_test = X_viz.copy()
    print("Shape of X test is: ", (X_test.shape))
    Y_test = fnc.f(X_test.T).reshape(-1, 1)

    return X_viz, X_test, Y_test


def visualize_predictions():
    # Training parameters
    NUM_TRAINING_POINTS = 100

    # Start to train for each individual:
    for fnc_name, fnc, active_d in FNC_TUPLES:
        print("Working on function: ", fnc)
        # Generate test set

        print("Weights are: ")
        print(fnc.W)

        X_train, Y_train = get_training_set(NUM_TRAINING_POINTS, fnc)
        X_viz, X_test, Y_test = get_test_set(80, fnc)

        do_plotting(fnc_name, X_viz, Y_test)

        # Train REMBO (this is separate, bcs does not have a GP interface
        # for name, algo in [("REMBO", initialize_rembo(fnc._domain))]:
        #     print("Training ", name)
        #     algo.add_data({
        #         'x': X_train,
        #         'y': Y_train
        #     })
        # 
        #     print("Predicting test data")
        #     Y_hat = algo.gp.mean(X_test)
        #     do_plotting_real_vs_gaussian(fnc_name + "_" + name, X_viz, Y_test, Y_hat)
        # 
        # 
        # for name, algo in [
        #     ("TRIPATHY", initialize_tripathy(fnc._domain)),
        #     ("BORING", initialize_boring(fnc._domain)),
        # ]:
        #     print("Training ", name)
        #     algo.add_data(X_train, Y_train)
        # 
        #     print("Predicting test data")
        #     Y_hat = algo.mean(X_test)
        #     do_plotting_real_vs_gaussian(fnc_name + "_" + name, X_viz, Y_test, Y_hat)


if __name__ == "__main__":
    print("Starting to visualize functions")
    visualize_predictions()
    sys.exit(0)
