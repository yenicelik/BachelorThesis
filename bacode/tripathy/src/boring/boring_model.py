from GPy.util.linalg import dpotrs
from febo.utils import get_logger

import numpy as np

import sys

# sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/bacode/tripathy")
sys.path.append("/cluster/home/yedavid/BachelorThesis/tripathy/")
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/bacode")

from febo.models import ConfidenceBoundModel
from febo.models.model import ModelConfig
from febo.models.gpy import GPRegression
from febo.utils.config import ConfigField, assign_config, config_manager
import GPy
from scipy.linalg import lapack
from scipy.optimize import minimize

from GPy.kern.src.rbf import RBF
from GPy.kern import Matern32

# from GPy.kern.src.sde_matern import Matern32

logger = get_logger('model')


class BoringModelConfig(ModelConfig):
    """
    * kernels: List of kernels
    * noise_var: noise variance

    """
    # kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
    noise_var = ConfigField(0.1)
    calculate_gradients = ConfigField(False, comment='Enable/Disable computation of gradient on each update.')
    optimize_bias = ConfigField(False)
    optimize_var = ConfigField(False)
    bias = ConfigField(0)
    _section = 'src.tripathy__'


config_manager.register(BoringModelConfig)

# def optimize_gp(experiment):
#     experiment.algorithm.f.gp.kern.variance.fix()
#     experiment.algorithm.f.gp.optimize()
#     print(experiment.algorithm.f.gp)

from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer
from bacode.tripathy.src.boring.generate_orthogonal_basis import generate_orthogonal_matrix_to_A


@assign_config(BoringModelConfig)
class BoringGP(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.

    """

    def create_kernels(self, active_dimensions, passive_dimensions, first=False, k_variance=None, k_lengthscales=None):

        # Use tripathy kernel here instead, because it includes W in it's stuff

        active_kernel = Matern32(
            input_dim=active_dimensions,
            variance=1. if k_variance is None else k_variance,
            lengthscale=1.5 if k_lengthscales is None else k_lengthscales,  # 0.5,
            ARD=True,
            active_dims=np.arange(active_dimensions),
            name="active_subspace_kernel"
        )

        self.kernel = active_kernel

        if first:  # TODO: need to change this back!

            # Now adding the additional kernels:
            for i in range(passive_dimensions):
                cur_kernel = RBF(
                    input_dim=1,
                    variance=2.,
                    lengthscale=0.5,  # 0.5,
                    ARD=False,
                    active_dims=[active_dimensions + i],
                    name="passive_subspace_kernel_dim_" + str(i)
                )

                self.kernel += cur_kernel

    def create_gp(self):

        self.gp = GPRegression(
            input_dim=self.domain.d,
            kernel=self.kernel,
            noise_var=0.01,
            calculate_gradients=False
        )

        # Let the GP take over datapoints from the datasaver!
        X = self.datasaver_gp.X
        Y = self.datasaver_gp.Y
        # Apply the Q transform if it was spawned already!
        if self.Q is not None:
            X = np.dot(X, self.Q)
        if self.Q is not None:
            assert X.shape[1] >= 2, ("Somehow, Q was not projected!", X.shape, 2) # TODO: change this back to ==!
        self.gp.set_XY(X, Y)
        self._update_cache()

    def create_gp_and_kernels(self, active_dimensions, passive_dimensions, first=False, k_variance=None,
                              k_lengthscales=None):
        self.create_kernels(active_dimensions, passive_dimensions, first=first, k_variance=k_variance,
                            k_lengthscales=k_lengthscales)
        self.create_gp()

    # From here on, it's the usual functions
    def __init__(self, domain, always_calculate=False):
        super(BoringGP, self).__init__(domain)

        # passive projection matrix still needs to be created first!
        # print("WARNING: CONFIG MODE IS: ", config.DEV)
        self.burn_in_samples = 101  # 101 # 102
        self.recalculate_projection_every = 101
        self.active_projection_matrix = None
        self.passive_projection_matrix = None
        self.Q = None

        # some other parameters that are cached
        self.t = 0

        # Setting the datasaver (infrastructure which allows us to save the data to be projected again and again)
        placeholder_kernel = RBF(
            input_dim=self.domain.d
        )
        self.datasaver_gp = GPRegression(
            input_dim=self.domain.d,
            kernel=placeholder_kernel,
            noise_var=0.01,
            calculate_gradients=False
        )

        # Create a new kernel and create a new GP
        self.create_gp_and_kernels(self.domain.d, 0, first=True)  # self.domain.d - 2

        # Some post-processing
        self.kernel = self.kernel.copy()
        self._woodbury_chol = np.asfortranarray(
            self.gp.posterior._woodbury_chol)  # we create a copy of the matrix in fortranarray, such that we can directly pass it to lapack dtrtrs without doing another copy
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()
        self._Y = np.empty(shape=(0, 1))
        self._bias = self.config.bias
        self.always_calculate = always_calculate

    @property
    def beta(self):
        return np.sqrt(np.log(self.datasaver_gp.X.shape[0]))

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
        return self.gp  # GPRegression(self.domain.d, self.kernel, noise_var=self.config.noise_var, calculate_gradients=self.config.calculate_gradients)

    def add_data(self, x, y):
        """
        Add a new function observation to the GPs.
        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        """
        self.i = 1 if not ("i" in dir(self)) else self.i + 1
        # print("Add data ", self.i)
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        self.set_data(x, y, append=True)

    # TODO: check if this is called anyhow!
    def optimize(self):
        self._update_beta()

    def _update_cache(self):
        # if not self.config.calculate_gradients:
        self._woodbury_chol = np.asfortranarray(self.gp.posterior._woodbury_chol)
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()

        self._update_beta()

    def _optimize_bias(self):
        self._bias = minimize(self._bias_loss, self._bias, method='L-BFGS-B')['x'].copy()
        self._set_bias(self._bias)
        logger.info(f"Updated bias to {self._bias}")

    def _bias_loss(self, c):
        # calculate mean and norm for new bias via a new woodbury_vector
        new_woodbury_vector, _ = dpotrs(self._woodbury_chol, self._Y - c, lower=1)
        K = self.gp.kern.K(self.gp.X)
        mean = np.dot(K, new_woodbury_vector)
        norm = new_woodbury_vector.T.dot(mean)
        # loss is least_squares_error + norm
        return np.asscalar(np.sum(np.square(mean + c - self._Y)) + norm)

    def _set_bias(self, c):
        self._bias = c
        self.gp.set_Y(self._Y - c)
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()

    def _update_beta(self):
        logdet = self._get_logdet()
        logdet_priornoise = self._get_logdet_prior_noise()
        self._beta = np.sqrt(2 * np.log(1 / self.delta) + (logdet - logdet_priornoise)) + self._norm()

    def _optimize_var(self):
        # fix all parameters
        for p in self.gp.parameters:
            p.fix()

        if self.gp.kern.name == 'sum':
            for part in self.gp.kern.parts:
                part.variance.unfix()
        else:
            self.gp.kern.variance.unfix()
        self.gp.optimize()
        if self.gp.kern.name == 'sum':
            values = []
            for part in self.gp.kern.parts:
                values.append(np.asscalar(part.variance.values))
        else:
            values = np.asscalar(self.gp.kern.variance.values)

        logger.info(f"Updated prior variance to {values}")
        # unfix all parameters
        for p in self.gp.parameters:
            p.unfix()

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

        assert not np.isnan(x).all(), ("X is nan at some point!", x)

        if self.config.calculate_gradients or True:
            # In the other case, projection is done in a subfunction
            if self.Q is not None:
                x = np.dot(x, self.Q)
            mean, var = self.gp.predict_noiseless(x)
        else:
            mean, var = self._raw_predict(x)

        return mean + self._bias, var

    def mean_var_grad(self, x):
        # TODO: should this be here aswell?
        # TODO: check, that this is not actually used for new AND saved gp's!
        if self.Q is not None:
            x = np.dot(x, self.Q)
        return self.gp.predictive_gradients(x)

    def var(self, x):
        return self.mean_var(x)[1]

    def mean(self, x):
        return self.mean_var(x)[0]

    def set_data(self, X, Y, append=True):

        # First of all, save everything in the saver GP
        if append:
            X = np.concatenate((self.datasaver_gp.X, X), axis=0)
            Y = np.concatenate((self.datasaver_gp.Y, Y), axis=0)  # Should be axis=0
        self.datasaver_gp.set_XY(X, Y)

        # Now, save everything in the other GP but with a projected X value
        #
        X = self.datasaver_gp.X
        Y = self.datasaver_gp.Y

        # Do our optimization now
        if self.burn_in_samples == self.i or self.always_calculate:  # (self.i >= self.burn_in_samples and self.i % self.recalculate_projection_every == 1) or
            import time
            start_time = time.time()
            print("Adding data: ", self.i)

            optimizer = TripathyOptimizer()

            # TODO: the following part is commented out, so we can test, if the function works well if we give it the real matrix!
            self.active_projection_matrix, sn, l, s, d = optimizer.find_active_subspace(X, Y)
            #
            # print("BORING sampled the following active matrix: ")
            # print(self.active_projection_matrix)
            #
            passive_dimensions = max(self.domain.d - d, 0)
            # passive_dimensions = 1 # TODO: take out this part!
            # # passive_dimensions = 0
            #
            # Generate A^{bot} if there's more dimensions
            if passive_dimensions > 0:
                self.passive_projection_matrix = generate_orthogonal_matrix_to_A(self.active_projection_matrix,
                                                                                 passive_dimensions)
            else:
                self.passive_projection_matrix = None
            #
            # print("BORING sampled the following passive matrix: ")
            # print(self.passive_projection_matrix)

            # d = 2
            # self.active_projection_matrix = np.asarray([
            #     [-0.31894555, 0.78400512, 0.38970008, 0.06119476, 0.35776912],
            #     [-0.27150973, 0.066002, 0.42761931, -0.32079484, -0.79759551]
            # ]).T
            # s = 1.
            # l = 1.5
            # passive_dimensions = 0

            # Create Q by concatenateing the active and passive projections
            if passive_dimensions > 0:
                self.Q = np.concatenate((self.active_projection_matrix, self.passive_projection_matrix),
                                        axis=1)
            else:
                self.Q = self.active_projection_matrix

            assert not np.isnan(self.Q).all(), ("The projection matrix contains nan's!", self.Q)

            print("BORING sampled the following matrix: ")
            print(self.Q)

            assert d == self.active_projection_matrix.shape[1]

            self.create_gp_and_kernels(
                active_dimensions=d,
                passive_dimensions=passive_dimensions,
                first=True,
                k_variance=s,
                k_lengthscales=l)

            print("Projection matrix is: ", self.Q.shape)
            print("Dimensions found are: ", d)
            print("Active projection matrix is ", self.active_projection_matrix.shape)
            print("How many datapoints do we have in the kernel?", self.gp.X.shape)
            print("How many datapoints do we have in the kernel?", self.datasaver_gp.X.shape)

            print("--- %s seconds ---" % (time.time() - start_time))

        # if self.i > self.burn_in_samples:
        #     assert self.Q is not None, "After the burning in, self.Q is still None!"

        # Add the points to the newly shapes GP!
        if (self.i < self.burn_in_samples or self.Q is None) and (not self.always_calculate):
            print("Still using the old method!")
            self.gp.set_XY(X, Y)
        else:
            print("We use the dot product thingy from now on!")
            Z = np.dot(X, self.Q)
            print("Old shape: ", X.shape)
            print("New shape: ", Z.shape)
            self.gp.set_XY(Z, Y)

        print("Added data: ", self.i)
        print("Datasave has shape: ", self.datasaver_gp.X.shape)
        print("Another shape: ", self.gp.X.shape)

        self.t = X.shape[0]
        self._update_cache()

    def _raw_predict(self, Xnew):
        m, n = Xnew.shape

        # Need to project Xnew here?
        # if self.Q is not None:
        #     Xnew = np.dot(Xnew, self.Q)
        #     assert Xnew.shape[1] == self.Q.reshape(self.Q.shape[0], -1).shape[1], ("Shapes are wrong: ", Xnew.shape, self.Q.shape)
        # else:
        #     assert Xnew.shape[1] == self.domain.d, ("Shapes are wrong when we have no Q!", Xnew.shape, self.domain.d)

        if not hasattr(self.kernel, 'parts'):  # TODO: take this out?
            mu, var = self._raw_predict_given_kernel(Xnew, self.kernel)
            # print("Using the cool values! ")
        else:
            mu = np.zeros((Xnew.shape[0], 1))
            var = np.zeros((Xnew.shape[0], 1))
            for kernel in self.kernel.parts:
                cur_mu, cur_var = self._raw_predict_given_kernel(Xnew, kernel)
                assert not np.isnan(cur_mu).all(), ("nan encountered for mean!", cur_mu)
                assert not np.isnan(cur_var).all(), ("nan encountered for var!", cur_var)
                mu += cur_mu
                var += cur_var

        assert not np.isnan(mu).all(), ("nan encountered for mean!", mu)
        assert not np.isnan(var).all(), ("nan encountered for mean!", var)

        assert mu.shape == (m, 1), ("Shape of mean is different! ", mu.shape, (m, 1))
        assert var.shape == (m, 1), ("Shape of variance is different! ", var.shape, (m, 1))

        return mu, var

    def _raw_predict_given_kernel(self, Xnew, kernel):
        Kx = kernel.K(self._X, Xnew)
        mu = np.dot(Kx.T, self._woodbury_vector)

        if len(mu.shape) == 1:
            mu = mu.reshape(-1, 1)

        Kxx = kernel.Kdiag(Xnew)
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
