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
    # dim = ConfigField(2, comment='subspace dimension')
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
        # assert not np.isnan(Z).all(), ("One of the optimized values over the acquisition function it nan!", Z)
        # print(Z)
        return -self.gp.ucb(Z)

    def _next(self):
        z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)
        return z_ucb