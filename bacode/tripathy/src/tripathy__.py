from GPy.util.linalg import dpotrs
from febo.utils import get_logger

import numpy as np

import sys
# sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/bacode/tripathy")
# sys.path.append("/cluster/home/yedavid/BachelorThesis/tripathy/")
# sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/bacode")


from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.models.gpy import GPRegression
from febo.utils.config import ConfigField, assign_config, config_manager
import GPy
from scipy.linalg import lapack
from scipy.optimize import minimize

from bacode.tripathy.src.bilionis_refactor.t_optimization_functions import t_ParameterOptimizer

logger = get_logger('tripathy')

from febo.utils import locate, get_logger


class TripathyGPConfig(ModelConfig):
    """
    * kernels: List of kernels
    * noise_var: noise variance

    """
    # kernels = ConfigField([('GPy.kern.Matern32', {'variance': 1., 'lengthscale': 1.5, 'ARD': True})])
    noise_var = ConfigField(0.01)
    calculate_gradients = ConfigField(False, comment='Enable/Disable computation of gradient on each update.')
    optimize_bias = ConfigField(False)
    optimize_var = ConfigField(False)
    bias = ConfigField(0)
    _section = 'src.tripathy__'


config_manager.register(TripathyGPConfig)

from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from GPy.kern import Matern32
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer


@assign_config(TripathyGPConfig)
class TripathyGP(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.

    """

    def create_new_kernel(self, active_d, W=None, variance=None, lengthscale=None):
        print("Creating a new kernel!")
        self.kernel = TripathyMaternKernel(
            real_dim=self.domain.d,
            active_dim=active_d,
            W=W,
            variance=variance,
            lengthscale=lengthscale
        )

    def create_new_gp(self, noise_var=None):
        # Take over data from the old GP, if existent
        print("Creating a new gp!")
        self.gp = GPRegression(
            self.domain.d,
            self.kernel,
            noise_var=noise_var if noise_var is not None else self.config.noise_var,
            calculate_gradients=self.config.calculate_gradients
        )

    def create_new_gp_and_kernel(self, active_d, W, variance, lengtscale, noise_var):
        self.create_new_kernel(
            active_d=active_d,
            W=W,
            variance=variance,
            lengthscale=lengtscale
        )
        self.create_new_gp(
            noise_var=noise_var
        )

    def __init__(self, domain, calculate_always=False):
        super(TripathyGP, self).__init__(domain)

        print("Starting tripathy model!")
        self.gp = None

        self.active_d = None
        self.W_hat = None
        self.variance = None
        self.lengthscale = None
        self.noise_var = None

        self.create_new_gp_and_kernel(
            active_d=self.domain.d if self.active_d is None else self.active_d,
            W=np.eye(self.domain.d) if self.active_d is None else self.W,
            variance=1.0 if self.active_d is None else self.variance,
            lengtscale=1.5 if self.active_d is None else self.lengthscale,
            noise_var=None if self.active_d is None else self.noise_var,
        )

        # number of data points
        self.t = 0
        self.i = 0
        self.kernel = self.kernel.copy()
        self._woodbury_chol = np.asfortranarray(
            self.gp.posterior._woodbury_chol)  # we create a copy of the matrix in fortranarray, such that we can directly pass it to lapack dtrtrs without doing another copy
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()
        self._Y = np.empty(shape=(0, 1))
        self._beta = 2
        self._bias = self.config.bias
        self.calculate_always = calculate_always

    # Obligatory values
    @property
    def beta(self):
        return self._beta

    @property
    def scale(self):
        if self.gp.kern.name == 'sum':
            return sum([part.variance for part in self.gp.kern.parts])
        else:
            return np.sqrt(self.gp.kern.inner_kernel.variance)

    @property
    def bias(self):
        return self._bias

    def _get_gp(self):
        return self.gp

    def add_data(self, x, y):
        """
        Add a new function observation to the GPs.
        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        assert x.shape[1] == self.domain.d, "Input dimension is not the one of the domain!"

        self.i += 1

        self.set_data(x, y, append=True)

    def optimize(self):
        self._update_beta()

    def _update_cache(self):
        # if not self.config.calculate_gradients:
        self._woodbury_chol = np.asfortranarray(self.gp.posterior._woodbury_chol)
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()

        self._update_beta()

    def _update_beta(self):
        logdet = self._get_logdet()
        logdet_priornoise = self._get_logdet_prior_noise()
        self._beta = np.sqrt(2 * np.log(1 / self.delta) + (logdet - logdet_priornoise)) + self._norm()

    def _get_logdet(self):
        return 2. * np.sum(np.log(np.diag(self.gp.posterior._woodbury_chol)))

    def _get_logdet_prior_noise(self):
        return self.t * np.log(self.gp.likelihood.variance.values)

    def mean_var(self, x):
        """Recompute the confidence intervals form the GP.
        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        x = np.atleast_2d(x)

        if self.config.calculate_gradients and False: # or True:
            mean, var = self.gp.predict_noiseless(x)
        else:
            mean, var = self._raw_predict(x)

        return mean + self._bias, var

    def var(self, x):
        return self.mean_var(x)[1]

    def mean(self, x):
        return self.mean_var(x)[0]

    # TODO: Implement the thing finder in here!
    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.gp.X, X))
            Y = np.concatenate((self.gp.Y, Y))

        if self.i % 200 == 100 or self.calculate_always:

            optimizer = TripathyOptimizer()
            self.W_hat, self.noise_var, self.lengthscale, self.variance, self.active_d = optimizer.find_active_subspace(X, Y)
            #
            # print("Optimized parameters are: ")
            # print(self.noise_var)
            # print(self.lengthscale)
            # print(self.variance)
            # print(self.active_d)

            print("Found parameters are: ")
            print("W: ", self.W_hat)
            print("noise_var: ", self.noise_var)
            print("lengthscale: ", self.lengthscale)
            print("variance: ", self.variance)


            # print("Adapting projection! ")
            #
            # # TODO: use optimizer instead of real projection matrix?
            # self.active_d = 2
            # self.W_hat = np.asarray([
            #     [-0.31894555, 0.78400512, 0.38970008, 0.06119476, 0.35776912],
            #     [-0.27150973, 0.066002, 0.42761931, -0.32079484, -0.79759551]
            # ]).T
            # self.variance = np.asarray(5.)
            # self.lengthscale = np.ones((self.active_d,)) * 1.5
            #
            # self.noise_var = None

            # For the sake of creating a kernel with new dimensions!
            self.create_new_gp_and_kernel(
                active_d=self.active_d,
                W=self.W_hat,
                variance=self.variance,
                lengtscale=self.lengthscale,
                noise_var=self.noise_var
            )

            # Recalculate the kernel and gp with optimized values
            ######
            # Optimize over all parameters
            # self.parameter_optimizer = t_ParameterOptimizer(
            #     self.W_hat,
            #     self.kernel,
            #     X,
            #     Y
            # )
            #
            # self.variance, self.lengthscale, new_noise = self.parameter_optimizer.optimize_s_sn_l(
            #     self.config.noise_var,
            #     self.variance,
            #     self.lengthscale
            # )
            #
            # self.create_new_gp_and_kernel(
            #     active_d=self.active_d,
            #     W=self.W_hat,
            #     variance=self.variance,
            #     lengtscale=self.lengthscale,
            #     noise_var=self.noise_var
            # )

            self._set_data(X, Y)


            # print("Optimized parameters are: ")
            # print(self.variance)
            # print(self.lengthscale)
            # print(new_noise)
            # print("Printing the entire gp now")
            # print(self.gp)
            ####

        else:

            self._set_data(X, Y)

    def _set_data(self, X, Y):
       self.gp.set_XY(X, Y)
       self.t = X.shape[0]
       self._update_cache()

    def _raw_predict(self, Xnew):

        Kx = self.kernel.K(self._X, Xnew)
        mu = np.dot(Kx.T, self._woodbury_vector)

        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)

        Kxx = self.kernel.Kdiag(Xnew)
        tmp = lapack.dtrtrs(self._woodbury_chol, Kx, lower=1, trans=0, unitdiag=0)[0]
        var = (Kxx - np.square(tmp).sum(0))[:, None]
        return mu, var

    def _norm(self):
        norm = self._woodbury_vector.T.dot(self.gp.kern.K(self.gp.X)).dot(self._woodbury_vector)
        return np.asscalar(np.sqrt(norm))

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict[
            'gp']  # remove the gp from state dict to allow pickling. calculations are done via the cache woodbury/cholesky
        return self_dict