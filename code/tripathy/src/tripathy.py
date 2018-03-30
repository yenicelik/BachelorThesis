
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

# class TripathyConfig(Config):
#     num_layers = ConfigField(4, comment="Number of layers used")
#     num_neurons = ConfigField([100,100,50, 50], comment="Number of units in each layer.")
#     learning_rate = ConfigField('deep.learning_rate', comment="Function providing the learning rate.")
#     _section = 'deep.model'

def optimize_gp(experiment):
    experiment.algorithm.f.gp.kern.variance.fix()
    experiment.algorithm.f.gp.optimize()
    print(experiment.algorithm.f.gp)

# TODO: how to update the kernel values?
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
        self.t = 0

        # TODO: handle this kernel part somehow!
        # TODO: put the active dimension etc. in a different file later
        # TODO: make active and real dimension adaptable/altereable (i guess you just create a new GPRegression?)
        self.kernel = TripathyMaternKernel(2, 2)
        self.gp = GPRegression(d, self.kernel, noise_var=self.config.noise_var)
        # number of data points

    ###############################
    #      SAMPLING FUNCTIONS     #
    ###############################
    def sample_sn(self):
        return 2.

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