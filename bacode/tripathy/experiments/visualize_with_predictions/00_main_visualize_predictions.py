# The following object includes the function, and domain information for each tuple
from febo.environment.benchmarks import ParabolaEmbedded2D, DecreasingSinusoidalEmbedded5D, CamelbackEmbedded5D

from bacode.tripathy.experiments.angle_optimization.utils import visualize_angle_given_W_array
from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel

import numpy as np

from bacode.tripathy.src.bilionis_refactor.t_optimizer import run_two_step_optimization

from bacode.tripathy.src.rembo.rembo_algorithm import RemboAlgorithm
from bacode.tripathy.src.boring.boring_algorithm import BoringGP
from bacode.tripathy.src.tripathy__ import TripathyGP

import sys

FNC_TUPLES = [
    ["Parabola-2D->1D", ParabolaEmbedded2D(), 1],
    # ["Camelback-5D->2D", CamelbackEmbedded5D(), 2],
    # ["Sinusoidal-5D->2D", DecreasingSinusoidalEmbedded5D(), 2]
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
    X_train = np.random.rand(points, fnc.d)
    X_train = (X_train * fnc._domain.range) + fnc._domain._center
    Y_train = fnc.f(X_train.T).reshape(-1, 1)
    print("Shape of X, Y: ", (X_train.shape, Y_train.shape))
    return X_train, Y_train

def get_test_set(points, fnc):
    # Xs: np.ndarray = np.meshgrid(
    #     *[
    #         np.arange(fnc._domain.l[i], fnc._domain.l[i], get_stepsize(fnc._domain)) for i in range(fnc._domain.d)
    #     ]
    # fnc._domain.l,
    # fnc_domain.
    #     np.arange(lower[0], upper[0], step_size[0]),
    # np.arange(lower[1], upper[1], step_size[1]),
    # )
    # X_test = np.vstack([X_ele.flatten() for X_ele in Xs]).T
    # print("Shape of X test is: ", (X_test.shape, Y_test.shape))
    pass

def get_stepsize(domain, dim):
    # Per dimension, spawn
    pass



def visualize_predictions():
    # Training parameters
    NUM_TRAINING_POINTS = 100

    # Start to train for each individual:
    for name, fnc, active_d in FNC_TUPLES:
        # Generate test set

        X_train, Y_train = get_training_set(NUM_TRAINING_POINTS, fnc)

        # Train REMBO (this is separate, bcs does not have a GP interface
        for name, algo in [("REMBO", initialize_rembo( fnc._domain) )]:
            print("Training ", name)
            algo.add_data({
                'x': X_train,
                'y': Y_train
            })

            print("Predicting test data")



        for name, algo in [
            ("TRIPATHY", initialize_tripathy( fnc._domain )),
            ("BORING", initialize_boring( fnc._domain )),
        ]:
            print("Training ", name)
            algo.add_data(X_train, Y_train)



if __name__ == "__main__":
    print("Starting to visualize functions")
    visualize_predictions()
    sys.exit(0)

