
import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_optimization_functions import t_WOptimizer
from febo.environment.benchmarks.functions import Rosenbrock
from febo.environment.benchmarks.functions import Camelback

from GPy.models.gp_regression import GPRegression


class Metrics(object):
    """
        We will use different metrics to check if our algorithm can successfully
        approximate the hidden matrix.
    """

    def __init__(self, sampling_points, seed=42):
        self.samples = 10
        np.random.seed(seed)

        self.tol_mean_diff = 1e-6 # TODO: set this to an ok value

    # TODO: visualization of embedded function
    # TODO: use utils - cartesian
    # TODO: optimizers cadidate GridOptimizer
    # TODO: plot(grid, f(grid))
    # TODO: plot gamma function
    # TODO: implement x^2 in 1D
    # TODO: simulate the real projection by 37-39

    def mean_difference_points(self, fnc, fnc_hat, A, A_hat, X):
        """
            ∀x in real_dim. E_x [ f(A x) - f(A_hat x) ] < tolerance
        :param fnc:
        :param A:
        :param A_hat:
        :return:
        """
        # TODO: change f to f_hat (from gaussian process)
        # TODO:

        assert A.shape == A_hat.shape, str(A.shape, A_hat.shape)

        # gp_reg = GPRegression(X, Y, kernel, sn)
        # y_hat = gp_reg.predict(X)
        # y = np.dot(X, A)


        X = np.random.rand(self.samples, A.shape[0])


        t1 = fnc( np.dot(X, A).T )
        t2 = fnc( np.dot(X, A_hat).T )

        print("Difference is: ")
        print(t1 - t2)

        return np.mean( np.abs(t1 - t2) ) < self.tol_mean_diff


    def projects_into_same_original_point(self, A, A_hat):
        """
            ∀x in real_dim. | (A A.T x) - (A_hat A_hat.T x) | < tolerance
        :param A:
        :param A_hat:
        :return:
        """
        # TODO: somehting is really funky here!
        assert A.shape == A_hat.shape, str((A.shape, A_hat.shape))
        assert not np.equal(A, A_hat)
        X = np.random.rand(self.samples, A.shape[0])

        t1 = np.dot(A.T, X.T)
        t1 = np.dot(A, t1).T

        t2 = np.dot(A_hat.T, X.T)
        t2 = np.dot(A_hat, t2).T

        out = []
        for i in range(self.samples):
            diff = np.abs(t1[i,:] - t2[i,:])
            print(t1[i,:], t2[i,:])
            truth_val = np.mean(diff) < self.tol_mean_diff
            out.append(truth_val)

        assert len(out) == self.samples
        return all(out)


class TestMatrixRecovery(object):
    """
        We hide a function depending on a matrix A within a higher dimension.
        We then test if our algorithm can successfully approximate/find this matrix
        (A_hat is the approximated one).

        More specifically, check if:

        f(A x) = f(A_hat x)
    """

    def init(self):

        self.real_dim = 2
        self.active_dim = 1
        self.no_samples = 25
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Hide the matrix over here!
        if self.real_dim == 3 and self.active_dim == 2:
            self.function = Camelback()
            self.real_W = np.asarray([
                [0, 1],
                [1, 0],
                [0, 0]
            ])
        elif self.real_dim == 2 and self.active_dim == 1:
#            self.function =
            self.real_W = np.asarray([
                [0],
                [1],
            ])
        else:
            assert False, "W was not set!"

        self.sn = 2.

        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T)

        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

        # We create the following kernel just to have access to the sample_W function!
        # TripathyMaternKernel(self.real_dim)

        self.tries = 1
        self.max_iter = 20
        self.m = 50

        self.metrics = Metrics(self.no_samples)

    def test_if_function_is_found(self):
        self.init()

        print("Real matrix is: ", self.real_W)

        all_tries = []
        for i in range(self.tries):
            # Initialize random guess
            W_hat = self.kernel.sample_W()

            # Find a good W!
            for i in range(self.max_iter):
                W_hat = self.w_optimizer.optimize_stiefel_manifold(W_hat, self.m)

            print("Difference to real W is: ", (W_hat - self.real_W))

            assert W_hat.shape == self.real_W.shape
            res = self.metrics.mean_difference_points(self.function._f, self.real_W, W_hat)
            all_tries.append(res)

        print(all_tries)

        assert np.asarray(all_tries).any()

    def test_if_hidden_matrix_is_found_multiple_initializations(self):
        self.init()

        print("Real matrix is: ", self.real_W)

        all_tries = []

        for i in range(self.tries):
            # Initialize random guess
            W_hat = self.kernel.sample_W()

            # Find a good W!
            for i in range(self.max_iter):
                W_hat = self.w_optimizer.optimize_stiefel_manifold(W_hat, self.m)

            print("Difference to real (AA.T) W is: ", (W_hat - self.real_W))

            assert W_hat.shape == self.real_W.shape
            assert not (W_hat == self.real_W).all()
            res = self.metrics.projects_into_same_original_point(self.real_W, W_hat)
            all_tries.append(res)

        assert True in all_tries

        # Check if projection is correct