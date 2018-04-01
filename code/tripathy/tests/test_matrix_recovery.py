
import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_optimization_functions import t_WOptimizer

class Metrics(object):
    """
        We will use different metrics to check if our algorithm can successfully
        approximate the hidden matrix.
    """

    def __init__(self, sampling_points, seed=42):
        self.samples = 1000
        np.random.seed(seed)

        self.tol_mean_diff = 1e-6 # TODO: set this to an ok value

    def mean_difference_points(self, fnc, A, A_hat):
        """
            ∀x in real_dim. E_x [ f(A x) - f(A_hat x) ] < tolerance
        :param fnc:
        :param A:
        :param A_hat:
        :return:
        """
        assert A.shape == A_hat.shape, str(A.shape, A_hat.shape)

        X = np.random.rand(self.samples, A.shape)

        t1 = fnc( np.dot(X, A) )
        t2 = fnc( np.dot(X, A_hat) )

        return np.mean( np.abs(t1 - t2) ) < self.tol_mean_diff


    def projects_into_same_original_point(self, A, A_hat):
        """
            ∀x in real_dim. | (A A.T x) - (A_hat A_hat.T x) | < tolerance
        :param A:
        :param A_hat:
        :return:
        """
        assert A.shape == A_hat.shape, str(A.shape, A_hat.shape)
        X = np.random.rand(self.samples, A.shape)

        t1 = np.dot(A.T, X.T)
        t1 = np.dot(A, t1)

        t2 = np.dot(A_hat.T, X.T)
        t2 = np.dot(A_hat, t2)

        return np.mean(np.abs(t1 - t2)) < self.tol_mean_diff


        # We randomly sample points, and check if they project to the same sapce


class TestMatrixRecovery(object):
    """
        We hide a function depending on a matrix A within a higher dimension.
        We then test if our algorithm can successfully approximate/find this matrix
        (A_hat is the approximated one).

        More specifically, check if:

        f(A x) = f(A_hat x)
    """

    def init(self):

        # Choose an arbitrary test function
        self.m = 50

        self.real_dim = 3
        self.active_dim = 2
        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.sn = 2.

        self.W = np.random.rand(self.real_dim, self.active_dim)

        self.X = np.random.rand(self.no_samples, self.real_dim)
        self.Y = np.random.rand(self.no_samples)

        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

        # We create the following kernel just to have access to the sample_W function!
        # TripathyMaternKernel(self.real_dim)

        self.A = self.kernel.sample_W()
        self.max_iter = 10000

    def test_if_hidden_matrix_is_found(self):
        self.init()

        # Start from random orthogonal matrix A
        self.A_hat = self.kernel.sample_W()

        print("Real matrix is: ", self.A)
        self.w_optimizer.optimize_stiefel_manifold(self.A_hat, self.m)

