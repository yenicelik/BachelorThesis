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


class TripathyMaternKernel(Kern):

    """
    A kernel of the following form:
        .. math::
            k(x, x') = k_0(Wx, Wx')
    """

    def __init__(self, d, variance, lengthscale):

        # TODO: decide what file to put these values (this file is probably good, just check how to pass around stuff)
        self.W = self.sample_random_orth_matrix(self.real_dim, self.active_d)
        self.l = self.sample_lengthscale()
        self.s = self.sample_variance()

        # TODO: find a way to change internal variables within the following object!
        self.inner_kernel = Matern32(self.active_dims, self.s, self.l, ARD=True)

        # TODO: incorporate a way to include W as the kernel-parameter (e.g. call parent function, where x = W.T x)
        # TODO: incorporate a way to easily switch out all kernel-parameters (variance and lengthscale)

        # TODO: overwrite the kernel parameters!

        super(TripathyMaternKernel, self).__init__(d, self.s, self.l)

    ###############################
    #       SETTER FUNCTIONS      #
    ###############################
    def set_W(self, W):
        self.W = W

    def set_l(self, l):
        self.l = l
        self.inner_kernel.lengthscale = Param("lengthscale", l, Logexp())

    def set_s(self, s):
        self.s = s
        self.inner_kernel.variance = Param("variance", s, Logexp())

    ###############################
    #      SAMPLING FUNCTIONS     #
    ###############################
    def sample_W(self, real_dim, active_dim):
        """
        Returns: An orthogonal matrix
        """
        A = np.zeros((real_dim, active_dim), dtype=np.float64)
        for i in range(real_dim):
            for j in range(active_dim):
                A[i, j] = np.random.normal(0, 1)
        Q, R = np.linalg.qr(A)
        assert (np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[0])))
        return Q

    def sample_variance(self):
        return 0.1

    def sample_lengthscale(self):
        return np.ones((self.real_dim,)) * 0.2

    ###############################
    #        KERNEL-FUNCTIONS     #
    ###############################
    def K(self, X1, X2):
        """
        Overriding the kernel method
        :param X:
        :param X2:
        :return:
        """
        Z1 = np.dot(X1, self.W)
        Z2 = np.dot(X2, self.W)
        return self.inner_kernel.K(Z1, Z2)
        # TODO: VERY IMPORTANT! does this calculate the gram-matrix (i.e. does this work also for matrix-inputs?

    def Kdiag(self, X):
        Z = np.dot(X, self.W)
        return self.inner_kernel.Kdiag(Z)
