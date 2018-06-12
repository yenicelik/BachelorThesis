from pprint import pprint

from febo.models.gpy import GPRegression

from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer
from bacode.tripathy.src.rembo.utils import sample_orthogonal_matrix
from bacode.tripathy.src.boring.utils import get_quantiles

import numpy as np
from febo.algorithms import Algorithm, AlgorithmConfig
from febo.environment import ContinuousDomain
from febo.models import GP
from febo.optimizers import ScipyOptimizer
from febo.utils.config import ConfigField, config_manager, assign_config


class BoringConfig(AlgorithmConfig):
    dim = ConfigField(2, comment='subspace dimension')
    optimize_every = ConfigField(40,
                                 comment='adding how many datapoints will lead to identifying the active and passive subspace?')


config_manager.register(BoringConfig)

# Imports for this specific function!
from bacode.tripathy.src.boring.boring_model import BoringGP
from bacode.tripathy.src.boring.generate_orthogonal_basis import generate_orthogonal_matrix_to_A
import time


@assign_config(BoringConfig)
class BoringAlgorithm(Algorithm):

    def add_data(self, data):
        """
        Add a new function observation to the GPs.
        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        """

        x = data['x']
        y = data['y']

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        self.data_counter += 1

        self.gp.add_data(x, y)

    def initialize(self, **kwargs):
        """
            self.domain carries the higher-dimensional domain
            self.config.dim carries the lower-dimensional domain
        :param kwargs:
        :return:
        """
        super(BoringAlgorithm, self).initialize(**kwargs)
        # self.model = BoringGP(self.domain)

        self.data_counter = 0

        self.jitter = 0.01  # TODO: something to be rather taken from the config file

        self.optimizer = ScipyOptimizer(self.domain)
        self.gp = BoringGP(self.domain)  # TODO: remove this!

    #############################
    #   Acquisition functions   #
    #############################
    def ucb_acq_function(self, Z):
        assert not np.isnan(Z).all(), ("One of the optimized values over the acquisition function it nan!", Z)
        return -self.gp.ucb(Z)

    def _next(self):
        z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)
        return z_ucb

    # def add_data(self, data):
    #     """
    #     Add a new function observation to the GPs.
    #     Parameters
    #     ----------
    #     x: 2d-array
    #     y: 2d-array
    #     """
    #
    #     x = data['x']
    #     y = data['y']
    #
    #     x = np.atleast_2d(x)
    #     y = np.atleast_2d(y)
    #
    #     self.data_counter += 1
    #
    #     # self.set_data(x, y, append=True)
    #
    #     self.gp.add_data(x, y, append=True)
    #
    # def set_data(self, X, Y, append=True):
    #     # self.model.set_data(X, Y, append)
    #     if append:
    #         X = np.concatenate((self.gp.gp.X, X))
    #         Y = np.concatenate((self.gp.gp.Y, Y))
    #
    #     self.gp.set_data(X, Y, append=False)
    #     # gp.set_XY(X, Y)  # TODO: need to do this for each individual components once it is successful
    #     self.t = X.shape[0]
    #
    # def initialize(self, **kwargs):
    #     """
    #         self.domain carries the higher-dimensional domain
    #         self.config.dim carries the lower-dimensional domain
    #     :param kwargs:
    #     :return:
    #     """
    #     super(BoringAlgorithm, self).initialize(**kwargs)
    #     # self.model = BoringGP(self.domain)
    #
    #     self.data_counter = 0
    #
    #     self.jitter = 0.01  # TODO: something to be rather taken from the config file
    #
    #     self.optimizer = ScipyOptimizer(self.domain)
    #     self.gp = GP(self.domain) # TODO: remove this!
    #
    #
    # #############################
    # #   Acquisition functions   #
    # #############################
    # def ucb_acq_function(self, Z):
    #     return -self.gp.ucb(Z)
    #     # return -self.model.ucb(Z)
    #
    # def _next(self):
    #     z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)
    #     return z_ucb

#
# @assign_config(BoringConfig)
# class BoringAlgorithm(Algorithm):
#
#     #############################
#     # Tripathy specific actions #
#     #############################
#     def set_new_kernel(self, d, W=None, variance=None, lengthscale=None):
#         self.active_kernel = TripathyMaternKernel(
#             real_dim=self.domain.d,
#             active_dim=d,
#             W=W,
#             variance=variance,
#             lengthscale=lengthscale
#         )
#
#     def set_new_gp(self, noise_var=None):
#         self.active_gp = GPRegression(
#             input_dim=self.domain.d,
#             kernel=self.active_kernel,
#             noise_var=noise_var if noise_var else 2.,
#             calculate_gradients=True
#         )
#
#     def set_new_active_gp_and_kernel(self, d, W, variance, lengthscale, noise_var):
#         self.set_new_kernel(d, W, variance, lengthscale)
#         self.set_new_gp(noise_var)
#         # Finally, set the new acquisition function
#         # TODO: Also re-initialize the activation function?
#         # self.acquisition_object = AcquisitionMPI(self.active_gp, self.domain)
