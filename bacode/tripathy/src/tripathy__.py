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

logger = get_logger('model')

class TripathyGPConfig(ModelConfig):
    """
    * kernels: List of kernels
    * noise_var: noise variance

    """
    # kernels = ConfigField([('GPy.kern.RBF', {'variance': 2., 'lengthscale': 0.2 , 'ARD': True})])
    noise_var = ConfigField(0.01)
    calculate_gradients = ConfigField(False, comment='Enable/Disable computation of gradient on each update.')
    optimize_bias = ConfigField(False)
    optimize_var = ConfigField(False)
    bias = ConfigField(0)
    _section = 'src.tripathy__'

config_manager.register(TripathyGPConfig)

from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer

@assign_config(TripathyGPConfig)
class TripathyGP(ConfidenceBoundModel):
    """
    Base class for GP optimization.
    Handles common functionality.

    """

    def set_new_kernel(self, d, W=None, variance=None, lengthscale=None):
        self.kernel = TripathyMaternKernel(
            real_dim=self.domain.d,
            active_dim=d,
            W=W,
            variance=variance,
            lengthscale=lengthscale
        )

    def set_new_gp(self, noise_var=None):
        self.gp = GPRegression(
            input_dim=self.domain.d,
            kernel=self.kernel,
            noise_var=noise_var if noise_var else 2., # self.config.noise_var,
            calculate_gradients=self.config.calculate_gradients
        )

    def set_new_gp_and_kernel(self, d, W, variance, lengthscale, noise_var):
        self.set_new_kernel(d, W, variance, lengthscale)
        self.set_new_gp(noise_var)

    def __init__(self, domain, calculate_always=False):
        super(TripathyGP, self).__init__(domain)

        self.optimizer = TripathyOptimizer()

        # TODO: d is chosen to be an arbitrary value rn!
        self.set_new_gp_and_kernel(2, None, None, None, None)

        # number of data points
        self.t = 0
        self.kernel = self.kernel.copy()
        self._woodbury_chol = np.asfortranarray(self.gp.posterior._woodbury_chol)  # we create a copy of the matrix in fortranarray, such that we can directly pass it to lapack dtrtrs without doing another copy
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()
        self._Y = np.empty(shape=(0,1))
        self._beta = 2
        self._bias = self.config.bias
        self.calculate_always = calculate_always

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
        return GPRegression(self.domain.d, self.kernel, noise_var=self.config.noise_var, calculate_gradients=self.config.calculate_gradients)

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
        new_woodbury_vector,_= dpotrs(self._woodbury_chol, self._Y - c, lower=1)
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
        self._beta = np.sqrt(2 * np.log(1/self.delta) + (logdet - logdet_priornoise)) + self._norm()

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
        return 2.*np.sum(np.log(np.diag(self.gp.posterior._woodbury_chol)))

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

        if self.config.calculate_gradients or True:
            mean,var = self.gp.predict_noiseless(x)
        else:
            mean,var = self._raw_predict(x)

        return mean + self._bias, var

    def mean_var_grad(self, x):
        return self.gp.predictive_gradients(x)

    def var(self, x):
        return self.mean_var(x)[1]

    def mean(self, x):
        return self.mean_var(x)[0]

    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.gp.X, X))
            Y = np.concatenate((self.gp.Y, Y))

        # Do our optimization now
        if self.i % 100 == 99 or self.calculate_always:
            import time
            start_time = time.time()
            print("Adding data: ", self.i)

            # TODO: UNCOMMENT THE FOLLOWING LINE AGAIN!
            # This is just to check if tripathy conforms with the other version
            # W_hat, sn, l, s, d = self.optimizer.find_active_subspace(X, Y)
            d = 2
            W_hat = np.asarray([
                [-0.31894555, 0.78400512, 0.38970008, 0.06119476, 0.35776912],
                [-0.27150973, 0.066002, 0.42761931, -0.32079484, -0.79759551]
            ]).T
            s = 1.
            l = 1.5
            sn = 2. # 0.01 #self.config.noise_var

            print("--- %s seconds ---" % (time.time() - start_time))

            # Overwrite GP and kernel values
            self.set_new_gp_and_kernel(d=d, W=W_hat, variance=s, lengthscale=l, noise_var=sn)

        # TODO: Should the following not come before the optimization?

        self.gp.set_XY(X, Y)
        self.t = X.shape[0]
        self._update_cache()

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

    def _raw_predict(self, Xnew):

        Kx = self.kernel.K(self._X, Xnew)
        mu = np.dot(Kx.T, self._woodbury_vector)

        if len(mu.shape)==1:
            mu = mu.reshape(-1,1)

        Kxx = self.kernel.Kdiag(Xnew)
        tmp = lapack.dtrtrs(self._woodbury_chol, Kx, lower=1, trans=0, unitdiag=0)[0]
        var = (Kxx - np.square(tmp).sum(0))[:,None]
        return mu, var

    def _raw_predict_covar(self, Xnew, Xcond):
        Kx = self.kernel.K(self._X, np.vstack((Xnew,Xcond)))
        tmp = lapack.dtrtrs(self._woodbury_chol, Kx, lower=1, trans=0, unitdiag=0)[0]

        n = Xnew.shape[0]
        tmp1 = tmp[:,:n]
        tmp2 = tmp[:,n:]

        Kxx = self.kernel.K(Xnew, Xcond)
        var = Kxx - (tmp1.T).dot(tmp2)

        Kxx_new = self.kernel.Kdiag(Xnew)
        var_Xnew = (Kxx_new - np.square(tmp1).sum(0))[:,None]
        return var_Xnew, var

    def _norm(self):
        norm = self._woodbury_vector.T.dot(self.gp.kern.K(self.gp.X)).dot(self._woodbury_vector)
        return np.asscalar(np.sqrt(norm))

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['gp'] # remove the gp from state dict to allow pickling. calculations are done via the cache woodbury/cholesky
        return self_dict
