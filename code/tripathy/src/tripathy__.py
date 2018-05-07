from GPy.util.linalg import dtrtrs, tdot
from febo.utils import locate

import math
import numpy as np

from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.models.gpy import GPRegression
from febo.utils.config import ConfigField, assign_config, config_manager
import GPy

from .t_kernel import TripathyMaternKernel
from .t_optimizer import TripathyOptimizer

from febo.utils import locate

import math
import numpy as np

from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.utils.config import ConfigField, assign_config

import sys
import numpy as np
from scipy.ndimage import rotate
import scipy.optimize._minimize
from GPy.kern.src.sde_matern import sde_Matern32

import math

from febo.models import Model, ConfidenceBoundModel



# class GPConfig(ModelConfig):
#     kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
#     noise_var = ConfigField(0.1)
#     _section = 'models.gp'



class GPConfig(ModelConfig):
    """
    * kernels: List of kernels
    * noise_var: noise variance

    """
    # kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
    # noise_var = ConfigField(0.1)
    # calculate_gradients = ConfigField(True, comment='Enable/Disable computation of gradient on each update.')
    # _section = 'models.gp'

config_manager.register(GPConfig)

def optimize_gp(experiment):
    experiment.algorithm.f.gp.kern.variance.fix()
    experiment.algorithm.f.gp.optimize()
    print(experiment.algorithm.f.gp)

@assign_config(GPConfig)
class TripathyGP(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.

    """
    def set_hyperparameters(self):
        self.noise_var = 0.1
        self.calculate_gradients = True

    def set_new_kernel_and_gp(self, d):
        self.kernel = TripathyMaternKernel(
            real_dim=self.domain.d,
            active_dim=d
        )

        self.gp = GPRegression(
            input_dim=self.domain.d,
            kernel=self.kernel,
            noise_var=self.noise_var,
            calculate_gradients=self.calculate_gradients
        )

    def __init__(self, domain):
        super(TripathyGP, self).__init__(domain)

        self.set_hyperparameters()

        self.set_new_kernel_and_gp(domain.d)

        self.t = 0
        self.kernel = self.kernel.copy()  # TODO: what do I need this thing?
        self._woodbury_chol = self.gp.posterior._woodbury_chol.copy()
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()

    @property
    def beta(self):
        return 2  # math.sqrt(math.log(max(self.t,2))) # TODO: we could use the theoretical bound calculated from the kernel matrix.

    def add_data(self, X, Y):
        """

        Args:
            x:
            y:

        Returns:

        """
        # print("Adding data to GP!!!")
        # logger.info(f'Adding new data of shape!! {self.x.shape}.')

        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        X = np.concatenate((self.gp.X, X))
        Y = np.concatenate((self.gp.Y, Y))
        self.gp.set_XY(X, Y)
        self.t = X.shape[0]

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
        self._woodbury_chol = self.gp.posterior._woodbury_chol.copy()
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()

        self.t += y.shape[1]

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP.
        Parameters
        ----------
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """

        self.gp.remove_last_data_point()

    def mean_var(self, x):
        """Recompute the confidence intervals form the GP.
        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        x = np.atleast_2d(x)
        # return self.gp.predict_noiseless(x)
        return self._raw_predict(x, self._X)

    def mean_var_grad(self, x):
        return self.gp.predictive_gradients(x)

    def var(self, x):
        return self.mean_var(x)[1]

    def predictive_var(self, X, X_cond):
        X = np.atleast_2d(X)
        X_cond = np.atleast_2d(X_cond)
        KXX = self.kernel.K(X_cond, X).reshape(-1, 1)
        # print(KXX, "KXX", KXX*KXX, (1 + self.var(X_cond)), )
        # print('pred_var', self.var(X) - KXX*KXX/(self.config.noise_var + self.var(X_cond)))
        return self.var(X) - KXX * KXX / (self.noise_var + self.var(X_cond))

    def mean(self, x):
        return self.mean_var(x)[0]

    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.gp.X, X))
            Y = np.concatenate((self.gp.Y, Y))
        self.gp.set_XY(X, Y)
        self.t = X.shape[0]

    def sample(self, X=None):
        class GPSampler:
            def __init__(self, X, Y, kernel, var):
                self.X = X
                self.Y = Y
                self.N = var * np.ones(shape=Y.shape)
                self.kernel = kernel
                self.m = GPy.models.GPHeteroscedasticRegression(self.X, self.Y, self.kernel)
                self.m['.*het_Gauss.variance'] = self.N

            def __call__(self, X):
                X = np.atleast_2d(X)
                sample = np.empty(shape=(X.shape[0], 1))

                # iteratively generate sample values for all x in x_test
                for i, x in enumerate(X):
                    sample[i] = self.m.posterior_samples_f(x.reshape((1, -1)), size=1)

                    # add observation as without noise
                    self.X = np.vstack((self.X, x))
                    self.Y = np.vstack((self.Y, sample[i]))
                    self.N = np.vstack((self.N, 0))

                    # recalculate model
                    self.m = GPy.models.GPHeteroscedasticRegression(self.X, self.Y, self.kernel)
                    self.m['.*het_Gauss.variance'] = self.N  # Set the noise parameters to the error in Y

                return sample

        return GPSampler(self.gp.X.copy(), self.gp.Y.copy(), self.kernel, self.gp.likelihood.variance)

    def _raw_predict(self, Xnew, pred_var, full_cov=False):

        Kx = self.kernel.K(pred_var, Xnew)
        mu = np.dot(Kx.T, self._woodbury_vector)

        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)

        if full_cov:
            raise NotImplementedError
            Kxx = self.kernel.K(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = Kxx - tdot(tmp.T)
            elif self._woodbury_chol.ndim == 3:  # Missing data
                var = np.empty((Kxx.shape[0], Kxx.shape[1], self._woodbury_chol.shape[2]))
                for i in range(var.shape[2]):
                    tmp = dtrtrs(self._woodbury_chol[:, :, i], Kx)[0]
                    var[:, :, i] = (Kxx - tdot(tmp.T))
            var = var
        else:
            Kxx = self.kernel.Kdiag(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = (Kxx - np.square(tmp).sum(0))[:, None]
            elif self._woodbury_chol.ndim == 3:  # Missing data
                var = np.empty((Kxx.shape[0], self._woodbury_chol.shape[2]))
                for i in range(var.shape[1]):
                    tmp = dtrtrs(self._woodbury_chol[:, :, i], Kx)[0]
                    var[:, i] = (Kxx - np.square(tmp).sum(0))
            var = var
        return mu, var

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict[
            'gp']  # remove the gp from state dict to allow pickling. calculations are done via the cache woodbury/cholesky
        return self_dict
#
# # class TripathyConfig(Config):
# #     num_layers = ConfigField(4, comment="Number of layers used")
# #     num_neurons = ConfigField([100,100,50, 50], comment="Number of units in each layer.")
# #     learning_rate = ConfigField('deep.learning_rate', comment="Function providing the learning rate.")
# #     _section = 'deep.model'
#
# # def optimize_gp(experiment):
# #     experiment.algorithm.f.gp.kern.variance.fix()
# #     experiment.algorithm.f.gp.optimize()
# #     print(experiment.algorithm.f.gp)
# #
# # # TODO: how to update the kernel values?
# # # TODO: Do the optimization here! Implement the specific functions somewhere else though!
# # # TOOD: This is because we have to optimize both the kernel-parameters, as well as the Gaussian-prior parameters! (s_n, s, l)
# # @assign_config(GPConfig)
# # class TripathyModel(ConfidenceBoundModel):
# #     """
# #     Base class for GP optimization.
# #     Handles common functionality.
# #     Parameters
# #     ----------
# #     """
# #
# #     def __init__(self, d):
# #         super(TripathyModel, self).__init__(d)
# #
# #         # Auxiliary parameters
# #         self.MAX_ITER = 50
# #         self.active_d = 2 # TODO: Active dimensions. Let this be a searched parameter at some later stage
# #         self.real_dim = d
# #         self.tau_max = 1e-3
# #
# #         # Toleranec parameters
# #         #  TODO: assign each of these tolerances to one value
# #         self.xtol = 1e-6
# #         self.ftol = 1e-6  # Assume this is the stiefel-manifold function
# #         self.gtol = 1e-12  # Assume this is the tau-function
# #
# #         # Parameters to be optimized
# #         self.sn = self.sample_sn()
# #
# #
# #         # Prior parameters
# #         self.prior_mean = 0
# #         self.t = 0
# #
# #         # TODO: handle this kernel part somehow!
# #         # TODO: put the active dimension etc. in a different file later
# #         # TODO: make active and real dimension adaptable/altereable (i guess you just create a new GPRegression?)
# #         self.kernel = TripathyMaternKernel(self.real_dim, self.active_d)
# #         self.gp = GPRegression(d, self.kernel, noise_var=self.config.noise_var)
# #         # number of data points
# #
# #         self.t_optimizer = TripathyOptimizer()
# #
# #     ###############################
# #     #      SAMPLING FUNCTIONS     #
# #     ###############################
# #     def sample_sn(self):
# #         return 2.
# #
# #     ###############################
# #     #    DATA ADDERS & REMOVERS   #
# #     ###############################
# #     def set_data(self, X, Y, append=True):
# #         if append:
# #             X = np.concatenate((self.gp.X, X))
# #             Y = np.concatenate((self.gp.Y, Y))
# #         self.gp.set_XY(X, Y)
# #         self.t = X.shape[0]
# #
# #     def add_data(self, x, y):
# #         self.add_data_point_to_gps(x,y)
# #
# #         print("Call 2-step-optimizer when t: ", self.t)
# #
# #         W = self.t_optimizer.run_two_step_optimization(self.kernel, self.config.noise_var, self.gp.X,
# #                                                        self.gp.Y)  # right now, we assume that we know the active subdimension!
# #         self.kernel.set_W(W)
# #
# #     def add_data_point_to_gps(self, x, y):
# #         """
# #         Add a new function observation to the GPs.
# #         Parameters
# #         ----------
# #         x: 2d-array
# #         y: 2d-array
# #         """
# #         x = np.atleast_2d(x)
# #         y = np.atleast_2d(y)
# #         self.gp.append_XY(x, y)
# #
# #         self.t = self.gp.X.shape[0]
# #
# #     def remove_last_data_point(self):
# #         """Remove the data point that was last added to the GP.
# #         Parameters
# #         ----------
# #             gp: Instance of GPy.models.GPRegression
# #                 The gp that the last data point should be removed from
# #         """
# #
# #         self.gp.remove_last_data_point()
# #
# #         self.t = self.gp.X.shape[0]
# #
# #     ###############################
# #     #        GP FUNCTIONS         #
# #     ###############################
# #     def mean_var(self, x):
# #         """Recompute the confidence intervals form the GP.
# #         Parameters
# #         ----------
# #         context: ndarray
# #             Array that contains the context used to compute the sets
# #         """
# #         return self.gp.predict_noiseless(x)
# #
# #     def mean_var_grad(self, x):
# #         return self.gp.predictive_gradients(x)
# #
# #     def var(self, x):
# #         return self.mean_var(x)[1]
# #
# #     def mean(self, x):
# #         # TODO: regardless of the mean value, the same regret is being returned!!!
# #         return self.mean_var(x)[0]
# #
# #     @property
# #     def beta(self):
# #         return math.sqrt(math.log(max(self.t,2))) # TODO: we could use the theoretical bound calculated from the kernel matrix.
