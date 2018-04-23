
import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_optimization_functions import t_WOptimizer
from febo.environment.benchmarks.functions import Rosenbrock
from febo.environment.benchmarks.functions import Camelback, Parabola
from src.t_optimizer import TripathyOptimizer
from src.t_loss import loss, dloss_dW, dK_dW, dloss_ds

from GPy.models.gp_regression import GPRegression


# TODO: check if the same value is attained as the maximum

class Metrics(object):
    """
        We will use different metrics to check if our algorithm can successfully
        approximate the hidden matrix.
    """

    def __init__(self, sampling_points, seed=42):
        self.samples = 10
        np.random.seed(seed)

        self.tol_mean_diff = 1e-3 # TODO: set this to an ok value
        self.tol_abs_diff = 1e-2

    # TODO: simulate the real projection by 37-39

    def mean_difference_points(self, fnc, fnc_hat, A, A_hat, X):
        """
            ∀x in real_dim. E_x [ f(A x) - f_hat(A_hat x) ] < tolerance
        :param fnc:
        :param fnc_hat: is the prediction function of the GPregression
        :param A:
        :param A_hat:
        :return:
        """

        assert A.shape == A_hat.shape, str(A.shape, A_hat.shape)
        assert X.shape[1] == A.shape[0], (X.shape, A.shape)

        # TODO: update the gaussian process with the new kernels parameters! (i.e. W_hat)

        Z = np.dot(X, A).T
        y = fnc( Z ).T

        # The projection of X to the subspace happens within the gaussian process (due to the kernel)
        y_hat = fnc_hat( X )[0]

        assert y_hat.shape == y.shape, (y_hat.shape, y.shape)

        print("MEAN_DIFF_POINTS: Difference is: ")
        print(np.sort(np.abs(y - y_hat))[:3])
        print(np.sort(np.abs(y - y_hat))[-3:])
        print("END OF MEAN_DIFF_POINTS")

        # Absolute error should be smaller than 1e-2
        # return np.max(y - y_hat) < self.tol_abs_diff

        return (np.abs( (y - y_hat) / y_hat) < self.tol_mean_diff).all()
        # return np.mean( np.abs(y - y_hat) ) < self.tol_mean_diff


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

class TestMatrixRecoveryNaive(object):

    def __init__(self):
        pass
        # This skips the test

    def init(self):
        self.real_dim = 3
        self.active_dim = 2

        self.no_samples = 20 # 50
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Parameters
        self.sn = 0.1
        self.W = self.kernel.sample_W()

        self.function = Camelback()
        self.real_W = np.asarray([
            [0, 1],
            [1, 0],
            [0, 0]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

        # [[0.9486833]
        #  [0.31622777]]

        self.X = np.random.rand(self.no_samples, self.real_dim)
        print(self.X.shape)
        Z = np.dot(self.X, self.real_W)
        print(Z.shape)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        self.no_tries = 2

    def test_visualize_augmented_sinusoidal_function(self):

        self.init()

        import os
        if not os.path.exists("./pics/camelback/"):
            os.makedirs("./pics/")
            os.makedirs("./pics/camelback/")

        #################################
        #     TRAIN THE W_OPTIMIZER     #
        #################################

        Opt = TripathyOptimizer()

        print("Real hidden matrix is: ", self.real_W)

        for j in range(self.no_tries):
            print("Try number : ", j)

            W_hat = self.kernel.sample_W()
            self.kernel.update_params(
                W=W_hat,
                s=self.kernel.inner_kernel.variance,
                l=self.kernel.inner_kernel.lengthscale
            )

            W_hat, sn, l, s = Opt.run_two_step_optimization(self.kernel, self.sn, self.X, self.Y)

            # Create the gp_regression function and pass in the predictor function as f_hat
            self.kernel.update_params(W=W_hat, l=l, s=s)
            gp_reg = GPRegression(self.X, self.Y, self.kernel, noise_var=sn)
            y_hat = gp_reg.predict(self.X)[0].squeeze()

            #################################
            #   END TRAIN THE W_OPTIMIZER   #
            #################################

            # Save the W just in case
            l = loss(
                self.kernel,
                W_hat,
                sn,
                s,
                l,
                self.X,
                self.Y
            )
            np.savetxt("./pics/camelback/Iter_" + str(j) + "__" + "Loss_" + str(l) + ".txt", W_hat)


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
        self.no_samples = 75
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
            self.function = Parabola()
            self.real_W = np.asarray([
                [1],
                [1],
            ])
            self.real_W = self.real_W / np.linalg.norm(self.real_W)
        else:
            assert False, "W was not set!"

        self.sn = 0.1

        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        self.w_optimizer = t_WOptimizer(
            self.kernel,
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

        # We create the following kernel just to have access to the sample_W function!
        # TripathyMaternKernel(self.real_dim)

        self.tries = 10
        self.max_iter =  1 # 150

        assert False

        self.metrics = Metrics(self.no_samples)

    def test_if_function_is_found(self):
        """
            Replace these tests by the actual optimizer function!
        :return:
        """
        self.init()

        print("Real matrix is: ", self.real_W)

        all_tries = []
        for i in range(self.tries):
            # Initialize random guess
            W_hat = self.kernel.sample_W()

            # Find a good W!
            for i in range(self.max_iter):
                W_hat = self.w_optimizer.optimize_stiefel_manifold(W_hat)

            print("Difference to real W is: ", (W_hat - self.real_W))

            assert W_hat.shape == self.real_W.shape
            self.kernel.update_params(
                W=W_hat,
                l=self.kernel.inner_kernel.lengthscale,
                s=self.kernel.inner_kernel.variance
            )

            # TODO: update the gaussian process with the new kernels parameters! (i.e. W_hat)

            # Create the gp_regression function and pass in the predictor function as f_hat
            gp_reg = GPRegression(self.X, self.Y, self.kernel, noise_var=self.sn)
            res = self.metrics.mean_difference_points(
                fnc=self.function._f,
                fnc_hat=gp_reg.predict,
                A=self.real_W,
                A_hat=W_hat,
                X=self.X
            )

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
                W_hat = self.w_optimizer.optimize_stiefel_manifold(W_hat)

            print("Difference to real (AA.T) W is: ", (W_hat - self.real_W))

            assert W_hat.shape == self.real_W.shape
            assert not (W_hat == self.real_W).all()
            res = self.metrics.projects_into_same_original_point(self.real_W, W_hat)
            all_tries.append(res)

        assert True in all_tries