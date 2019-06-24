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

from bacode.tripathy.src.bilionis_refactor.config import config
from bacode.tripathy.src.bilionis_refactor.t_optimization_functions import t_ParameterOptimizer

logger = get_logger('tripathy')

from febo.utils import locate, get_logger
import gc


class TripathyGPConfig(ModelConfig):
    """
    * kernels: List of kernels
    * noise_var: noise variance

    """
    # kernels = ConfigField([('GPy.kern.Matern32', {'variance': 1., 'lengthscale': 1.5, 'ARD': True})])
    noise_var = ConfigField(0.005)
    calculate_gradients = ConfigField(False, comment='Enable/Disable computation of gradient on each update.')
    optimize_bias = ConfigField(False)
    optimize_var = ConfigField(False)
    bias = ConfigField(0)
    _section = 'src.tripathy__'


config_manager.register(TripathyGPConfig)

from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from GPy.kern import Matern32, RBF
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer


@assign_config(TripathyGPConfig)
class TripathyGP(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.

    """

    # JOHANNES: Die folgenden drei funktionen
    # sind helper functions welche den kernel und gp neu-spawnend, da wir das später noch einmal machen werden müssen

    def create_new_kernel(self, active_d, variance, lengthscale):
        print("Creating a new kernel!")
        self.kernel = Matern32(
            input_dim=active_d,
            variance=variance,
            lengthscale=lengthscale,
            ARD=True,
            active_dims=np.arange(active_d),
            name="active_subspace_kernel"
        )
        print("Kernel is: ", self.kernel)

    def create_new_gp(self, active_d, noise_var):
        # Take over data from the old GP, if existent
        print("Creating a new gp!")
        self.gp = GPRegression(
            active_d,
            self.kernel,
            noise_var=noise_var,  # noise_var if noise_var is not None else self.config.noise_var,
            calculate_gradients=False  # self.config.calculate_gradients
        )

    def create_new_gp_and_kernel(self, active_d, variance, lengthscale, noise_var):
        self.create_new_kernel(
            active_d=active_d,
            variance=variance,
            lengthscale=lengthscale
        )
        self.create_new_gp(
            active_d=active_d,
            noise_var=noise_var
        )
        print("Got kernel: ")
        print(self.kernel)

    def __init__(self, domain, calculate_always=False):
        super(TripathyGP, self).__init__(domain)

        print("Starting tripathy model!")
        self.gp = None

        # Just for completeness
        # self.active_d = None
        # self.W_hat = None
        # self.variance = None
        # self.lengthscale = None
        # self.noise_var = None

        # DEFAULT
        self.W_hat = np.eye(self.domain.d)
        self.noise_var = 0.005
        # self.lengthscale = 0.08
        # self.variance = 0.1

        self.variance = 2.3555752428329177
        self.lengthscale = 4.8

        self.active_d = self.domain.d

        # PARABOLA
        # self.W_hat = np.asarray([[0.49969147, 0.1939272]]) # np.random.rand(self.d, 1).T
        # self.noise_var = 0.005
        # self.lengthscale = 6
        # self.variance = 2.5
        # self.active_d = 1

        # SINUSOIDAL
        # self.W_hat = np.asarray([
        #     [-0.41108301, 0.22853536, -0.51593653, -0.07373475, 0.71214818],
        #     [ 0.00412458, -0.95147725, -0.28612815, -0.06316891, 0.093885]
        # ])
        # self.noise_var = 0.005
        # self.lengthscale = 1.3
        # self.variance = 0.15
        # self.active_d = 2

        # CAMELBACK
        # self.W_hat = np.asarray([
        #     [-0.31894555, 0.78400512, 0.38970008, 0.06119476, 0.35776912],
        #     [-0.27150973, 0.066002, 0.42761931, -0.32079484, -0.79759551]
        # ])
        self.noise_var = 0.005
        self.lengthscale = 2.5
        self.variance = 1.0
        # self.active_d = 2

        self.create_new_gp_and_kernel(
            active_d=self.active_d,
            variance=self.variance,
            lengthscale=self.lengthscale,
            noise_var=self.noise_var
        )

        # JOHANNES: Damit wir später andere Matrizen zur  Projektion nutzen können,
        # speichere ich die Daten irgendwoch ab. Ich benutze die GP datenstruktur um
        # diese Daten abzuspeichern, einfach weil das einfacher ist

        # Create the datasaver GP
        placeholder_kernel = RBF(
            input_dim=self.domain.d
        )
        self.datasaver_gp = GPRegression(
            input_dim=self.domain.d,
            kernel=placeholder_kernel,
            noise_var=self.noise_var,
            calculate_gradients=False
        )

        # JOHANNES: Die folgenden Operationen habe ich übernommen aus dem febo GP

        # number of data points
        self.t = 0
        self.i = 0
        self._woodbury_chol = np.asfortranarray(
            self.gp.posterior._woodbury_chol)  # we create a copy of the matrix in fortranarray, such that we can directly pass it to lapack dtrtrs without doing another copy
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()
        self._Y = np.empty(shape=(0, 1))
        self._beta = 2
        self._bias = self.config.bias
        self.calculate_always = calculate_always

        self.optimizer = TripathyOptimizer()

    # JOHANNES: Die folgenden Operationen habe ich übernommen aus dem febo GP

    # Obligatory values
    @property
    def beta(self):
        return self._beta

    @property
    def scale(self):
        if self.gp.kern.name == 'sum':
            return sum([part.variance for part in self.gp.kern.parts])
        else:
            return np.sqrt(self.gp.kern.variance)

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
        self._X = self.gp.X.copy()  # TODO: should it be gp, or datasaver_gp?

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

        x = np.dot(x, self.W_hat.T)
        # print(x.shape)
        # print(self.W_hat.shape)
        assert x.shape[1] == self.active_d, ("The projected dimension does not equal to the active dimension: ", (self.active_d, x.shape))

        if self.config.calculate_gradients and False:  # or True:
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
            X = np.concatenate((self.datasaver_gp.X, X), axis=0)
            Y = np.concatenate((self.datasaver_gp.Y, Y), axis=0)
        self._set_datasaver_data(X, Y)

        if self.i % 500 == 100:

            # self.W_hat, self.noise_var, self.lengthscale, self.variance, self.active_d = self.optimizer.find_active_subspace(
            #     X, Y, load=False)
            # self.W_hat = self.W_hat.T

            # self.W_hat = np.asarray([
            #     [-0.41108301, 0.22853536, -0.51593653, -0.07373475, -0.71214818],
            #     [0.00412458, -0.95147725, -0.28612815, -0.06316891, -0.093885]
            # ])

            # self.W_hat = np.asarray([[-0.27219458],
            #  [0.08779054],
            #  [-0.28618016],
            #  [0.16803416],
            #  [0.89892623]]
            # ).T
            # self.lengthscale = [0.02285915]
            # self.variance = 2.5731190248142437
            # self.active_d = 1

            # self.W_hat = np.asarray([[0.39877165, 0.88585961],
            #                          [-0.23390389, 0.0992073],
            #                          [0.52560395, -0.03363714],
            #                          [0.0815202, 0.06316191],
            #                          [0.70948226, -0.44753746]]
            #                         ).T
            # self.W_hat = np.asarray([
            #     [-0.41108301, 0.22853536, -0.51593653, -0.07373475, -0.71214818],
            #     [0.00412458, -0.95147725, -0.28612815, -0.06316891, -0.093885]
            # ])
            # self.lengthscale = np.asarray([0.84471462, 4.75165394])
            # self.variance = 2.3555752428329177

            # self.W_hat = np.asarray([
            #     [-0.31894555, 0.78400512, 0.38970008, 0.06119476, 0.35776912],
            #     [-0.27150973, 0.066002, 0.42761931, -0.32079484, -0.79759551]
            # ])

            self.W_hat = np.asarray([[-0.13392005],
                   [0.0898743],
                   [-0.16831021],
                   [-0.02586708],
                   [0.97210627]]).T
            self.lengthscale = np.asarray([0.05191368])
            self.variance = 1.7733784121415521
            self.active_d = 1

            self.lengthscale = np.asarray([0.84471462])
            self.variance = 2.3555752428329177

            # self.W_hat = np.asarray([
            #     [-0.46554187, -0.36224966, 0.80749362],
            #     [0.69737806, -0.711918, 0.08268378]
            # ])

            # PARABOLA
            # self.W_hat = np.asarray([[0.49969147, 0.1939272]])
            #
            # self.noise_var = 0.005
            # self.lengthscale = 6
            # self.variance = 2.5
            # self.active_d = 1

            self.create_new_gp_and_kernel(
                active_d=self.active_d,
                variance=self.variance,
                lengthscale=self.lengthscale,
                noise_var=self.noise_var
            )

        #     self.W_hat = np.asarray([
        #         [-0.31894555, 0.78400512, 0.38970008, 0.06119476, 0.35776912],
        #         [-0.27150973, 0.066002, 0.42761931, -0.32079484, -0.79759551]
        #     ])
        #     self.noise_var = 0.005
        #     self.lengthscale = 2.5
        #     self.variance = 1.0
        #     self.active_d = 2
        #     print("Changed values")

        if self.i % 500 == 299:
            print("TRIPATHY :: Likelihood of the current GP is: ", self.gp.log_likelihood())

        Z = np.dot(X, self.W_hat.T)
        assert Z.shape[1] == self.active_d, (
            "Projected Z does not conform to active dimension", (Z.shape, self.active_d))
        self._set_data(Z, Y)

    def _set_datasaver_data(self, X, Y):
        self.datasaver_gp.set_XY(X, Y)

    def _set_data(self, X, Y):
        self.gp.set_XY(X, Y)
        self.t = X.shape[0]
        self._update_cache()

    def _raw_predict(self, Xnew):

        assert Xnew.shape[1] == self.active_d, ("Somehow, the input was not project")

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
