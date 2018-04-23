"""
    Create the kernel class here.
    The resulting kernel must inherit from the GPy kernel class
"""
import numpy as np
import scipy
from GPy.kern.src.kern import Kern
from GPy.kern.src.stationary import Matern32
from GPy.core.parameterization import Param
# from GPy.core.parameterization import param.Param
from paramz.transformations import Logexp

from GPy.kern.src.stationary import Matern32
from paramz.caching import Cache_this

class TripathyMaternKernel(Kern):

    """
    A kernel of the following form:
        .. math::
            k(x, x') = k_0(Wx, Wx')
    """

    def __init__(self, real_dim, active_dim, variance=None, lengthscale=None):

        assert(real_dim >= active_dim)

        self.real_dim = real_dim
        self.active_dim = active_dim

        # TODO: decide what file to put these values (this file is probably good, just check how to pass around stuff)
        # TODO: add these as priors
        self.W = self.sample_W()

        self.inner_kernel = Matern32(
            input_dim=self.active_dim,
            variance=self.sample_variance() if variance is None else variance,
            lengthscale=self.sample_lengthscale() if lengthscale is None else lengthscale,
            ARD=True)

        self.update_params(self.W, self.inner_kernel.lengthscale, self.inner_kernel.variance)

        # TODO: find a way to change internal variables within the following object!

        # TODO: incorporate a way to include W as the kernel-parameter (e.g. call parent function, where x = W.T x)

        # TODO: overwrite the kernel parameters!

        self.W_grad = np.zeros_like(self.W)

        super(TripathyMaternKernel, self).__init__(input_dim=self.real_dim, active_dims=None, name="TripathyMaternKernel")
        self.link_parameters(self.inner_kernel)

        # Add parameters
        # l = Param('outerKernel.lengthscale', self.inner_kernel.lengthscale)
        # self.link_parameters(l)

        # s = Param('outerKernel.variance', self.inner_kernel.variance)
        # self.link_parameters(s)
        # self.add_parameter(s, l)

    ###############################
    #       SETTER FUNCTIONS      #
    ###############################
    def update_params(self, W, l, s):
        self.set_l(l, True)
        self.set_s(s, True)
        # We will not include W as a parameter, as we want to call the derivatives etc. separatedly
        self.set_W(W, True)

    def set_W(self, W, safe=False):
        assert safe
        assert W.shape == (self.real_dim, self.active_dim)
        assert np.allclose( np.dot(W.T, W), np.eye(self.active_dim), atol=1.e-6), (W, np.dot(W.T, W), np.eye(self.active_dim))
        self.W = W
        self.parameters_changed()

    def set_l(self, l, safe=False):
        assert safe
        assert l.shape == (self.active_dim,)
        self.inner_kernel.lengthscale = l
        # TODO: do we have to link parameters here somehow? (with this kernel, NOT the inner kernel?)


    def set_s(self, s, safe=False):
        assert safe
        assert isinstance(s, float) or isinstance(s, Param), type(s)
        self.inner_kernel.variance = s

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
        assert np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]))
        assert Q.shape[0] == self.real_dim
        assert Q.shape[1] == self.active_dim
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

    ################################
    # INHERITING FROM INNER KERNEL #
    ################################
    def gradients_X(self, dL_dK, X, X2=None):
        Z = np.dot(X, self.W)
        Z2 = np.dot(X2, self.W) if X2 is not None else Z
        tmp = self.inner_kernel.gradients_X(dL_dK, Z, Z2)
        return np.einsum('ik,jk->ij', tmp, self.W)

    def update_gradients_full(self, dL_dK, X, X2):
        assert X2 is None
        Z = np.dot(X, self.W)
        # For all parameters that are also contained in the inner kernel,
        # i.e. lengthscale and variance
        self.inner_kernel.update_gradients_full(dL_dK, Z)

        # Setting the W gradients
        dL_dZ = self.inner_kernel.gradients_X(dL_dK, Z)
        self.W_grad = np.einsum('ij,ik->kj', dL_dZ, X)

