
from febo.utils import locate

import math
import numpy as np

from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.models.gpy import GPRegression
from febo.utils.config import ConfigField, assign_config


import sys
import numpy as np
from scipy.ndimage import rotate
import scipy.optimize._minimize
from GPy.kern.src.sde_matern import sde_Matern32

import math

from febo.models.gpy import GPRegression
from febo.models import Model, ConfidenceBoundModel

from .t_kernel import TripathyMaternKernel

class GPConfig(ModelConfig):
    kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
    noise_var = ConfigField(0.1)
    _section = 'models.gp'

from febo.utils.config import Config, ConfigField, assign_config
from febo.utils import locate

class TripathyConfig(Config):
    num_layers = ConfigField(4, comment="Number of layers used")
    num_neurons = ConfigField([100,100,50, 50], comment="Number of units in each layer.")
    learning_rate = ConfigField('deep.learning_rate', comment="Function providing the learning rate.")
    _section = 'deep.model'

@assign_config(TripathyConfig)
class Tripathy(ConfidenceBoundModel):

    def __init__(self):
        print("Initializing Tripathy model")


def optimize_gp(experiment):
    experiment.algorithm.f.gp.kern.variance.fix()
    experiment.algorithm.f.gp.optimize()
    print(experiment.algorithm.f.gp)

# TODO: Do the optimization here! Implement the specific functions somewhere else though!
# TOOD: This is because we have to optimize both the kernel-parameters, as well as the Gaussian-prior parameters! (s_n, s, l)
@assign_config(GPConfig)
class TripathyModel(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.
    Parameters
    ----------
    """

    def __init__(self, d):
        super(TripathyModel, self).__init__(d)

        # Auxiliary parameters
        self.MAX_ITER = 50
        self.active_d = 2 # TODO: Active dimensions. Let this be a searched parameter at some later stage
        self.real_dim = d
        self.tau_max = 1e-3

        # Toleranec parameters
        #  TODO: assign each of these tolerances to one value
        self.xtol = 1e-6
        self.ftol = 1e-6  # Assume this is the stiefel-manifold function
        self.gtol = 1e-12  # Assume this is the tau-function

        # Parameters to be optimized
        self.sn = self.sample_sn()


        # Prior parameters
        self.prior_mean = 0

        # TODO: handle this kernel part somehow!
        # the description of a kernel
        self.kernel = None
        for kernel_module, kernel_params in self.config.kernels:
            kernel_part = locate(kernel_module)(input_dim=d, **kernel_params)
            if self.kernel is None:
                self.kernel = kernel_part
            else:
                self.kernel += kernel_part

        # calling of the kernel
        self.gp = GPRegression(d, self.kernel, noise_var=self.config.noise_var)
        # number of data points
        self.t = 0

    ###############################
    #      SAMPLING FUNCTIONS     #
    ###############################
    def sample_sn(self):
        return 2.

    ###############################
    #     INHERITED FUNCTIONS     #
    ###############################


    ###############################
    #    DATA ADDERS & REMOVERS   #
    ###############################
    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.gp.X, X))
            Y = np.concatenate((self.gp.Y, Y))
        self.gp.set_XY(X, Y)
        self.t = X.shape[0]

    def add_data(self, x, y):
        self.add_data_point_to_gps(x,y)

#         # W update
#         #self.run_two_step_optimization_once(2)
#
#         # TODO. updatemodel
#         self.W = np.eye(self.real_dim)
#         #self.W = np.rot90(self.W)
#
#         # Create everything new, that needs to be updated, including the dimension, etc.
#         tmpX = self.gp.X
#         tmpY = self.gp.Y
#         self.kernel = sde_Matern32(input_dim=self.real_dim, variance=self.s, lengthscale=self.l, ARD=True)
#         self.gp = GPRegression(self.real_dim, self.kernel, noise_var=self.sn)
#         self.gp.set_XY(tmpX, tmpY)

    def add_data_point_to_gps(self, x, y):
        """
        Add a new function observation to the GPs.
        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        self.gp.append_XY(x, y)

        self.t = self.gp.X.shape[0]

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP.
        Parameters
        ----------
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """

        self.gp.remove_last_data_point()

        self.t = self.gp.X.shape[0]

    ###############################
    #        GP FUNCTIONS         #
    ###############################
    def mean_var(self, x):
        """Recompute the confidence intervals form the GP.
        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        return self.gp.predict_noiseless(x)

    def mean_var_grad(self, x):
        return self.gp.predictive_gradients(x)

    def var(self, x):
        return self.mean_var(x)[1]

    def mean(self, x):
        # TODO: regardless of the mean value, the same regret is being returned!!!
        return self.mean_var(x)[0]

    @property
    def beta(self):
        return math.sqrt(math.log(max(self.t,2))) # TODO: we could use the theoretical bound calculated from the kernel matrix.


#         # TODO: how to update the kernel values?
#         self.kernel = sde_Matern32(input_dim=d, variance=self.sn, lengthscale=self.l, ARD=True)
#         #self.kernel = RBF(input_dim=d, variance=self.sn, lengthscale=self.l, ARD=True)
#
#
#         # calling of the kernel
#         # TODO: change the kernel variance, and the kernel model!
#         # TODO: we are supposed to work on this kernel variance s_n! ### J
#         self.gp = GPRegression(d, self.kernel, noise_var=0.1)
#
#     #########################################
#     #                                       #
#     # All the derivative-specific functions #
#     #                                       #
#     #########################################
#     def dloss_W(self, W, sn, l, X, Y):
#
#         # TODO: create the gram matrix here! you need to take every possible combination
#
#         # TODO: this implementation seems to be complete garbage!
#         # TODO: at lesat the usual input to this function seems to be complete garbage!!
#
#         def _dK_dr(r):
#             # TODO: should be set sn as a class attribute?
#             return -3. * sn * r * np.exp(-np.sqrt(3.) * r)
#
#         # TODO: usually, a=X, and b=X. This means that the following d=0! this is very probably wrong!
#         # Calculate the term dK_dW
#
#         # TODO: ### J
#         # Instead of this, one should calculate the gram matrix of all such distances!
#         z1 = np.dot(X, W) #previously this way the first input "a"
#         z2 = np.dot(X, W) #previously, this was the seocnd input "b" (instead of X)
#         d = z1 - z2
#
#         f1 = _dK_dr(np.dot(d, d.T), sn)
#
#         f2_1 = 2 * np.divide(z1 - z2, l)
#         # TODO: this transpose feels terribly wrong!!
#         f2_1 = np.dot(f2_1, X.T) #previously, this way the first input "a.T"
#         f2_2 = 2 * np.divide(z2 - z1, l)
#         f2_2 = np.dot(f2_2, X.T) #previously, this was the second input "b.T"
#
#         f2 = f2_1 + f2_2
#         kernel_term = np.dot(f1, f2)
#
#         # Now we apply the chain rule, where the first element is the general derivative loss term
#         out = self.dloss_dparamfnc(W, sn, l, X, Y, kernel_term)
#         return out
#
#     def dloss_dsn(self, W, sn, l, X, Y):
#         pass
#
#     def dloss_l(self, W, sn, l, X, Y):
#         pass
#
#     def dloss_dparamfnc(self, W, sn, l, X, Y, param_derivative):
#
#         # Modify the kernel (just in case) before applying it
#         # TODO: this ARD is probably wrong!
#         self.kernel = sde_Matern32(input_dim=self.real_dim, variance=sn, lengthscale=l, ARD=True)
#
#         # Calculate the matrix we will later inver (K + sn I)
#         res_kernel = self._kernel(W=W)
#         K_sn = res_kernel + np.power(sn, 2) + np.eye(res_kernel.shape[0])
#
#         # Calculate the cholesky-decomposition for the matrix K_sn
#         L = np.linalg.cholesky(K_sn)
#         K_ss_inv = np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
#         # K_ss_inv = np.linalg.inv(K_sn)
#
#         # Calculate the displaced output
#         Y_hat = Y - self.prior_mean
#
#         # Calculate the first term
#         tmp_cholesky_inv = np.linalg.solve(L, Y_hat)
#         lhs_rhs = np.linalg.solve(L.T, tmp_cholesky_inv)
#         #        lhs_rhs = np.linalg.solve(K_sn, Y_hat)
#
#         s1 = np.dot(lhs_rhs, lhs_rhs.T)
#         s1 -= K_ss_inv
#         # TODO: how to correctly implement this kernel operation for each vector within X?
#
#         s1 = np.dot(s1, param_derivative)
#
#         # TODO: replace these values in other functions
#         #        kernel_derivative = self.dK_dW(X, X, sn, l, W)
#         #        s1 = np.dot(s1, kernel_derivative)
#         return -0.5 * np.matrix.trace(s1)
#
#     def dloss_dl(self, fix_W, fix_sn, l, X, Y):
#         pass
#
#     def dloss_dsn(self, fix_W, sn, fix_l, X, Y):
#         pass
#
