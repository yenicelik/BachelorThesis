"""
    This file contains functions which help us allow compare the swiss fel experiment to other methods.
"""
from bacode.tripathy.src.t_kernel import TripathyMaternKernel
from febo.models.gpy import GPRegression

def set_new_kernel(d, W=None, variance=None, lengthscale=None):
    return TripathyMaternKernel(
        real_dim=self.domain.d,
        active_dim=d,
        W=W,
        variance=variance,
        lengthscale=lengthscale
    )


def set_new_gp(self, noise_var=None):
    self.gp = GPRegression(
        input_dim=self.domain.d,
        kernel=self.kernel,
        noise_var=noise_var if noise_var else 2.,  # TODO: replace with config value!
        calculate_gradients=True  # TODO: replace with config value!
    )