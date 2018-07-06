
# The following object includes the function, and domain information for each tuple
from febo.environment.benchmarks import ParabolaEmbedded2D, DecreasingSinusoidalEmbedded5D, CamelbackEmbedded5D

from bacode.tripathy.experiments.angle_optimization.utils import visualize_angle_given_W_array
from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel

import numpy as np

from bacode.tripathy.src.bilionis_refactor.t_optimizer import run_two_step_optimization

FNC_TUPLES = [
    ["Parabola-2D->1D", ParabolaEmbedded2D(), 1],
    # ["Camelback-5D->2D", CamelbackEmbedded5D(), 2],
    # ["Sinusoidal-5D->2D", DecreasingSinusoidalEmbedded5D(), 2]
]

def visualize_angle_loss():

    # Training parameters
    NUM_TRAINING_POINTS = 100
    fnc_config = {
        "noise_var": 0.01
    }

    # Optimizer parameters
    local_config = {
        "M_l": 100,
        "m": 300,
        "n": 300,
        "losses": [],
        "leps": 1e-6
    }

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    local_config = dotdict(local_config)

    for name, fnc, active_d in FNC_TUPLES:
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
        all_Ws = run_two_step_optimization(
            local_config,
            t_kernel=t_kernel,
            sn=fnc_config['noise_var'],
            X=X_train,
            Y=Y_train,
            save_Ws=True
        )

        print(all_Ws)

        visualize_angle_given_W_array(fnc.W.T, all_Ws)

if __name__ == "__main__":
    print("Starting to visualize functions")
    visualize_angle_loss()