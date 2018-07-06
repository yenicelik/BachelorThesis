"""
    Here, we create the vanilla visualizations for all functions.
"""

import numpy as np

# TODO: the following does not work for anything that's not integer!
from febo.environment.benchmarks.functions import CamelbackEmbedded5D

# Generate the intervals at which we're visualizing each individual function (between the intervals of -1 and 1
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer

points_per_axis = 100.

def calculate_angle_between_two_matrices(A, B):
    M1 = np.dot(A, A.T)
    M2 = np.dot(B, B.T)
    diff = np.linalg.norm(M1 - M2, ord=2)
    return np.arcsin(diff)


###############################
#       5D to 2D VANILLA      #
###############################
def get_angle_between_real_and_found_W():

    # Set up common function information
    function_instance = CamelbackEmbedded5D()
    lower = np.asarray([-2., -1., -1., -1., -1.])
    upper = np.asarray([2., 1., 1., 1., 1.])
    drange = upper - lower
    dcenter = (upper + lower) / 2.

    # Generate the training samples
    X_train = (np.random.rand(500, function_instance.d) * drange) + dcenter
    Y_train = function_instance.f(X_train.T).reshape(-1, 1)
    print("Shape of X_train and Y_train: ", X_train.shape, Y_train.shape)

    # Optimize to find the W!
    real_W = function_instance.W.T

    # Iterate over samples of X, and see if the angle improves with increasing number of training samples!

    angles = []
    step = 26
    for i in range(step, X_train.shape[0], step):
        optimizer = TripathyOptimizer()
        print("Using first samples: ", i, X_train[:i,:].shape, Y_train[:i].shape)
        found_W, noise_var, lengthscale, variance, active_d = optimizer.find_active_subspace(
            X_train[:i,:],
            Y_train[:i]
        )
        cur_angle = calculate_angle_between_two_matrices(real_W, found_W)
        angles.append(cur_angle)

    print(angles)
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(angles))*step, angles)
    plt.ylabel('Angle of found subspace')
    plt.xlabel('Number of datapoints used for training')
    plt.show()



def main():

    print("Visualizing 5d to 2d plain")
    get_angle_between_real_and_found_W()

    print("Done!")


if __name__ == "__main__":
    print("Starting to visualize all functions")
    main()
