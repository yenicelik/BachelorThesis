from GPy.util.linalg import dtrtrs, tdot
from febo.utils import locate

import math
import numpy as np

from .src.t_kernel import TripathyMaternKernel
from .src.t_optimizer import TripathyOptimizer

from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.models.gpy import GPRegression
from febo.utils.config import ConfigField, assign_config, config_manager
import GPy


class GPConfig(ModelConfig):
    """
    * kernels: List of kernels
    * noise_var: noise variance

    """
    # kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
    noise_var = ConfigField(0.1)
    calculate_gradients = ConfigField(True, comment='Enable/Disable computation of gradient on each update.')
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

    def __init__(self, domain):
        super(TripathyGP, self).__init__(domain)

        # the description of a kernel
        # self.kernel = None
        # for kernel_module, kernel_params in self.config.kernels:
        #     kernel_part = locate(kernel_module)(input_dim=self.domain.d, **kernel_params)
        #     if self.kernel is None:
        #         self.kernel = kernel_part
        #     else:
        #         self.kernel += kernel_part

        # Let's do active-dim 2 for now. Later on change this!
        self.kernel = TripathyMaternKernel(
            real_dim=self.domain.d,
            active_dim=2
        )


        # calling of the kernel
        self.gp = GPRegression(self.domain.d, self.kernel, noise_var=self.config.noise_var, calculate_gradients=self.config.calculate_gradients)
        # number of data points
        self.t = 0
        self.kernel = self.kernel.copy()
        self._woodbury_chol = self.gp.posterior._woodbury_chol.copy()
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()

    @property
    def beta(self):
        return 2 # math.sqrt(math.log(max(self.t,2))) # TODO: we could use the theoretical bound calculated from the kernel matrix.



    def add_data(self, x, y):
        """

        Args:
            x:
            y:

        Returns:

        """
        self.add_data_point_to_gps(x,y)

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
        KXX = self.kernel.K(X_cond, X).reshape(-1,1)
        # print(KXX, "KXX", KXX*KXX, (1 + self.var(X_cond)), )
        # print('pred_var', self.var(X) - KXX*KXX/(self.config.noise_var + self.var(X_cond)))
        return self.var(X) - KXX*KXX/(self.config.noise_var + self.var(X_cond))

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

        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)

        if full_cov:
            raise NotImplementedError
            Kxx = self.kernel.K(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = Kxx - tdot(tmp.T)
            elif self._woodbury_chol.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],Kxx.shape[1], self._woodbury_chol.shape[2]))
                for i in range(var.shape[2]):
                    tmp = dtrtrs(self._woodbury_chol[:,:,i], Kx)[0]
                    var[:, :, i] = (Kxx - tdot(tmp.T))
            var = var
        else:
            Kxx = self.kernel.Kdiag(Xnew)
            if self._woodbury_chol.ndim == 2:
                tmp = dtrtrs(self._woodbury_chol, Kx)[0]
                var = (Kxx - np.square(tmp).sum(0))[:,None]
            elif self._woodbury_chol.ndim == 3: # Missing data
                var = np.empty((Kxx.shape[0],self._woodbury_chol.shape[2]))
                for i in range(var.shape[1]):
                    tmp = dtrtrs(self._woodbury_chol[:,:,i], Kx)[0]
                    var[:, i] = (Kxx - np.square(tmp).sum(0))
            var = var
        return mu, var

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['gp'] # remove the gp from state dict to allow pickling. calculations are done via the cache woodbury/cholesky
        return self_dict
