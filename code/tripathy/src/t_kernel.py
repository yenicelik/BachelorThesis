"""
    Create the kernel class here.
    The resulting kernel must inherit from the GPy kernel class
"""
import numpy as np
import scipy
from GPy.kern.src.kern import Kern
from GPy.kern.src.stationary import Matern32
from GPy.core.parameterization import Param
from paramz.transformations import Logexp

from GPy.kern.src.stationary import Matern32
from paramz.caching import Cache_this

class TripathyMaternKernel(Kern):

    """
    A kernel of the following form:
        .. math::
            k(x, x') = k_0(Wx, Wx')
    """

    def __init__(self, real_dim, active_dim):

        assert(real_dim >= active_dim)

        self.real_dim = real_dim
        self.active_dim = active_dim

        # TODO: decide what file to put these values (this file is probably good, just check how to pass around stuff)
        # TODO: add these as priors
        self.W = self.sample_W()

        # TODO: remove this line!!!
        self.W = np.eye(self.real_dim)

        self.l = self.sample_lengthscale()
        self.s = self.sample_variance()

        # TODO: find a way to change internal variables within the following object!

        # TODO: incorporate a way to include W as the kernel-parameter (e.g. call parent function, where x = W.T x)
        # TODO: incorporate a way to easily switch out all kernel-parameters (variance and lengthscale)

        # TODO: overwrite the kernel parameters!

        self.inner_kernel = Matern32(input_dim=self.active_dim, variance=self.s, lengthscale=self.l, ARD=True)

        super(TripathyMaternKernel, self).__init__(input_dim=self.real_dim, active_dims=None, name="TripathyMaternKernel")

    ###############################
    #       SETTER FUNCTIONS      #
    ###############################
    def set_W(self, W):
        assert(W.shape == (self.real_dim, self.active_dim))
        self.W = W

    def set_l(self, l):
        assert(l.shape == (self.active_dim,))
        self.l = l
        self.inner_kernel.lengthscale = Param("lengthscale", self.l, Logexp())

    def set_s(self, s):
        assert(isinstance(s, float))
        self.s = s
        self.inner_kernel.lengthscale = Param("variance", self.s, Logexp())

    ###############################
    #      SAMPLING FUNCTIONS     #
    ###############################
    def sample_W(self):
        """
        :return: An orthogonal matrix
        """
        A = np.zeros((self.real_dim, self.active_dim), dtype=np.float64)
        for i in range(self.real_dim):
            for j in range(self.active_dim):
                A[i, j] = np.random.normal(0, 1)
        Q, R = np.linalg.qr(A)
        assert (np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1])))
        return Q

    def sample_variance(self):
        """
        :return: A standard variance
        """
        return 0.1

    def sample_lengthscale(self):
        """
        :return: A standard length-scale
        """
        return np.ones((self.active_dim,)) * 0.2

    ###############################
    #        KERNEL-FUNCTIONS     #
    ###############################
    def K(self, X1, X2=None):
        """
        :param X1: A vector (or is a matrix allowed?)
        :param X2:
        :return:
        """
        assert X1.shape[1] == self.real_dim, (X1.shape, self.real_dim)
        if X2 is not None:
            assert X2.shape[1] == self.real_dim

        Z1 = np.dot(X1, self.W)
        Z2 = np.dot(X2, self.W) if X2 is not None else None
        return self.inner_kernel.K(Z1, Z2)
        # TODO: VERY IMPORTANT! does this calculate the gram-matrix (i.e. does this work also for matrix-inputs

    def Kdiag(self, X):
        """
        :param X:
        :return:
        """
        assert(X.shape[1] == self.real_dim)

        Z = np.dot(X, self.W)
        return self.inner_kernel.Kdiag(Z)

    # Oriented by
    # http: // gpy.readthedocs.io / en / deploy / _modules / GPy / kern / src / stationary.html  # Matern32.dK_dr
    def K_of_r(self, r):
        """
        :param r: The squared distance that one is to input
        :return:
        """
        return self.inner_kernel.K_of_r(r)

    ###############################
    #          DERIVATIVES        #
    ###############################
    def dK_dr(self, r):
        """
        :param r: The
        :return:
        """
        return self.inner_kernel.dK_dr(r)

    # TODO: we could use the function _squared_scaled_distance instead, which is inherited
    def r(self, x, y):
        """
        The squared distance function
        :param x:
        :param y:
        :return:
        """
        x_prime = np.dot(x, self.W)
        y_prime = np.dot(y, self.W)

        out = np.dot((x_prime-y_prime).T, (x_prime-y_prime))
        out = np.divide(out, np.power(self.l, 2))
        return out

    def dr_da(self, x, y):
        """
        The differential of the squared scaled distance w.r.t. the first input argument
        :param x:
        :param y:
        :return:
        """
        return 2 * np.divide(x-y, np.power(self.l, 2))

    def dr_db(self, x, y):
        """
        The differential of the squared scaled distance w.r.t. the second input argument
        :param x:
        :param y:
        :return:
        """
        return - 2 * np.divide(x-y, np.power(self.l, 2))

    ################################
    # INHERITING FROM INNER KERNEL #
    ################################
    def update_gradients_full(self, dL_dK, X, X2):
        """Set the gradients of all parameters when doing full (N) inference."""
        Z1 = np.dot(X, self.W)
        Z2 = np.dot(X2, self.W) if X2 is not None else None

        return self.inner_kernel.update_gradients_full(dL_dK, Z1, Z2)
