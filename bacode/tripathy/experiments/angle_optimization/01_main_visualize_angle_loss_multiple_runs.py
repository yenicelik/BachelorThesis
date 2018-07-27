
# The following object includes the function, and domain information for each tuple
from febo.environment.benchmarks import ParabolaEmbedded2D, DecreasingSinusoidalEmbedded5D, CamelbackEmbedded5D

from bacode.tripathy.experiments.angle_optimization.utils import visualize_angle_given_W_array, \
    calculate_angle_between_two_matrices, pad_2d_list, visualize_angle_array_stddev, visualize_loss_array_stddev
from bacode.tripathy.src.bilionis_refactor.config import config
from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel

import numpy as np
import random

from bacode.tripathy.src.bilionis_refactor.t_loss import loss
from bacode.tripathy.src.bilionis_refactor.t_optimizer import run_two_step_optimization

FNC_TUPLES = [
    # ["Parabola-2D->1D", ParabolaEmbedded2D(), 1],
    ["Camelback-5D->2D", CamelbackEmbedded5D(), 2],
    # ["Sinusoidal-5D->2D", DecreasingSinusoidalEmbedded5D(), 2]
]

def visualize_angle_loss():

    NUM_TRIES = 50

    # Training parameters
    NUM_TRAINING_POINTS = 100
    fnc_config = {
        "noise_var": 0.01
    }

    # Optimizer parameters
    # Following is ok for parabola
    # local_config = {
    #     "M_l": 100,
    #     "m": 1,
    #     "n": 1,
    #     "losses": [],
    #     "leps": 1e-16
    # }

    # Following is ok for camelback
    local_config = {
        "M_l": 1000,
        "m": 1,
        "n": 1,
        "losses": [],
        "leps": 1e-16
    }

    # local_config = {
    #     "M_l": 100,
    #     "m": 300,
    #     "n": 300,
    #     "losses": [],
    #     "leps": 1e-12
    # }

    # local_config = {
    #     "M_l": 10,
    #     "m": 10,
    #     "n": 10,
    #     "losses": [],
    #     "leps": 1e-6
    # }

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    local_config = dotdict(local_config)

    for name, fnc, active_d in FNC_TUPLES:

        if name == "Parabola-2D->1D":
            config['active_dimension'] = 1
            print("Active dimensions are at: ", config['active_dimension'])
        else:
            config['active_dimension'] = 2

        all_angles = []
        all_losses = []
        title = name + "_"+ str(NUM_TRIES) + "_" + str(random.randint(1,1000))

        # Run for a few times
        for n in range(NUM_TRIES):

            # Generate training points
            X_train = (
                    np.random.rand(NUM_TRAINING_POINTS, fnc.d) * fnc._domain.range
            ) + ((fnc._domain.u + fnc._domain.l) / 2.)
            Y_train = fnc.f(X_train.T).reshape(-1, 1)

            # Generate the kernel
            t_kernel = TripathyMaternKernel(
                real_dim=fnc.d,
                active_dim=active_d,
                W=None,
                variance=None,
                lengthscale=None
            )

            # Run the two-step-optimization on the respective function
            # Retrieve intermediate matrices from there
            all_Ws, best_config = run_two_step_optimization(
                local_config,
                t_kernel=t_kernel,
                sn=fnc_config['noise_var'],
                X=X_train,
                Y=Y_train,
                save_Ws=True,
                save_best_config=True
            )

            print(all_Ws)
            losses = [
                loss(
                    t_kernel,
                    W,
                    sn,
                    s,
                    l * np.ones((W.shape[1],)),
                    X_train,
                    Y_train
                ) for W, sn, l, s in all_Ws
            ]
            all_Ws = [x[0] for x in all_Ws]
            print("All losses are: ", local_config.losses)
            print("All        are: ", losses)

            # Calculate the angles
            angles = list(map(
                lambda x: calculate_angle_between_two_matrices(fnc.W.T, x),
                all_Ws
            ))

            all_angles.append(angles) # Creates a 2d array
            all_losses.append(losses) # Creates a 2d array

            # Check if the loss decreases after we receive the individual parameters

        # Pad the losses and arrays to the maximum size of the runs
        all_angles = pad_2d_list(all_angles)
        all_losses = pad_2d_list(all_losses)

        print(all_angles.shape)
        print(all_losses.shape)

        visualize_angle_array_stddev(all_angles, title=title)
        visualize_loss_array_stddev(all_losses, title=title)
        visualize_loss_array_stddev(all_losses, title=title, subtract_mean=True)

        # TODO: take max index for log-likelihood, and visualize angle and log-likelihood
        print("Retrieving array from: ", all_losses[:,-1].reshape(-1))
        max_index = np.argmax(all_losses[:,-1].reshape(-1))
        print("Best index is: ", max_index)
        print("Best found loss is: ")
        visualize_angle_array_stddev(all_angles, title=title, max_index=max_index)
        visualize_loss_array_stddev(all_losses, title=title, max_index=max_index)
        print("Done...")

if __name__ == "__main__":
    print("Starting to visualize functions")
    visualize_angle_loss()