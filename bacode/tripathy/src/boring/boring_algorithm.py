from pprint import pprint

from febo.models.gpy import GPRegression

from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer
from bacode.tripathy.src.rembo.utils import sample_orthogonal_matrix

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


def run_optimization():
    """
        Runs the optimization process, where we do the following:
            1.) Identify the active subspace, and find optimal parameters for the kernel
            2.) Generate an orthogonal basis to the active subspace, which represents the passive subspace
            3.) For each individual axes of the passive subspace, run an individual GP
    :return:
    """


@assign_config(BoringConfig)
class BoringAlgorithm(Algorithm):

    #############################
    # Tripathy specific actions #
    #############################
    def set_new_kernel(self, d, W=None, variance=None, lengthscale=None):
        self.active_kernel = TripathyMaternKernel(
            real_dim=self.domain.d,
            active_dim=d,
            W=W,
            variance=variance,
            lengthscale=lengthscale
        )

    def set_new_gp(self, noise_var=None):
        self.active_gp = GPRegression(
            input_dim=self.domain.d,
            kernel=self.active_kernel,
            noise_var=noise_var if noise_var else 2.,
            calculate_gradients=True
        )

    def set_new_active_gp_and_kernel(self, d, W, variance, lengthscale, noise_var):
        self.set_new_kernel(d, W, variance, lengthscale)
        self.set_new_gp(noise_var)

    ################
    # ADD NEW DATA #
    ################
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

        self.set_data(x, y, append=True)

    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.active_gp.gp.X, X))
            Y = np.concatenate((self.active_gp.gp.Y, Y))

        self.active_gp.gp.set_XY(X, Y) # TODO: need to do this for each individual components once it is successful
        self.t = X.shape[0]

        # Do our optimization now
        if self.data_counter % self.config.optimize_every == self.config.optimize_every - 1:
            print("Optimizing the algorithm rn")

            import time
            start_time = time.time()
            print("Adding data: ", self.data_counter)

            tripathy_optimizer = TripathyOptimizer()
            W_hat, sn, l, s, d = tripathy_optimizer.find_active_subspace(self.active_gp.gp.X, self.active_gp.gp.Y)

            print("--- %s seconds ---" % (time.time() - start_time))

            # Overwrite GP and kernel values
            self.set_new_active_gp_and_kernel(d=d, W=W_hat, variance=s, lengthscale=l, noise_var=sn)



    def initialize(self, **kwargs):
        """
            self.domain carries the higher-dimensional domain
            self.config.dim carries the lower-dimensional domain
        :param kwargs:
        :return:
        """
        super(BoringAlgorithm, self).initialize(**kwargs)

        self.data_counter = 0

        self.optimizer = ScipyOptimizer(self.domain)
        self.active_gp = GP(self.domain)

    #############################
    #   Acquisition functions   #
    #############################
    def mpi_acq_function(self, Z):
        """
            Is the maximum probability of improvement acquisition function
        :param Z:
        :return:
        """
        # return -self.gp
        raise NotImplementedError
        pass

    def ucb_acq_function(self, Z):
        return -self.active_gp.ucb(Z)

    def _next(self):
        z_ucb, _ = self.optimizer.optimize(self.ucb_acq_function)
        return z_ucb
