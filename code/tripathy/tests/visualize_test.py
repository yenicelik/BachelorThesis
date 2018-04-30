import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_optimization_functions import t_WOptimizer
from src.t_optimizer import TripathyOptimizer
from febo.environment.benchmarks.functions import Parabola, AugmentedSinusoidal, Rosenbrock, PolynomialKernel
from src.t_loss import loss, dloss_dW, dK_dW, dloss_ds
import matplotlib.pyplot as plt

"""
    ParallelLbfgs
"""

from febo.utils.utils import cartesian

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from GPy.models.gp_regression import GPRegression

class VisualizedTestingTau:

    def __init__(self):
        self.real_dim = 2
        self.active_dim = 1

        self.no_samples = 5
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Parameters
        self.sn = 2.
        self.W = self.kernel.sample_W()

        self.function = Parabola()
        self.real_W = np.asarray([
            [1],
            [1]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W).reshape((-1, 1))
        self.Y = self.function._f(Z.T).squeeze()

        self.w_optimizer = t_WOptimizer(
            self.kernel, # TODO: does the kernel take over the W?
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

        # Define the plotting variables
        self.tau_arr = np.linspace(0., self.w_optimizer.tau_max, 100)

    def visualize_tau_trajectory_for_random_W(self):
        """
            Visualize the trajectory of the gamma
            function against the loss function
            we have f(tau) = F( gamma(tau, W) )
        :return:
        """
        loss_arr = []

        # Sample a random W
        W_init = self.kernel.sample_W()
        for tau in self.tau_arr:
            print("New tau is: ", tau)
            W = self.w_optimizer._gamma(tau, W_init)
            loss_val = loss(
                self.kernel,
                W,
                self.sn,
                self.kernel.inner_kernel.variance,
                self.kernel.inner_kernel.lengthscale,
                self.X,
                self.Y.squeeze()
            )
            loss_arr.append(loss_val)

        print(loss_arr)

        plt.title("Tau vs Loss - Randomly sampled W")
        plt.scatter(self.tau_arr, loss_arr)
        plt.axis([min(self.tau_arr), max(self.tau_arr), min(loss_arr), max(loss_arr)])
        plt.show()

    def visualize_tau_trajectory_for_identity_W(self):
        """
            Visualize the trajectory of the gamma
            function against the loss function
            we have f(tau) = F( gamma(tau, W) )
        :return:
        """
        loss_arr = []

        # Sample a random W
        W_init = self.real_W
        for tau in self.tau_arr:
            print("New tau is: ", tau)
            W = self.w_optimizer._gamma(tau, W_init)
            loss_val = loss(
                self.kernel,
                W,
                self.sn,
                self.kernel.inner_kernel.variance,
                self.kernel.inner_kernel.lengthscale,
                self.X,
                self.Y.squeeze()
            )
            loss_arr.append(loss_val)

        print(loss_arr)

        plt.title("Tau vs Loss - Identity-similarsampled W")
        plt.scatter(self.tau_arr, loss_arr)
        plt.axis([min(self.tau_arr), max(self.tau_arr), min(loss_arr), max(loss_arr)])
        plt.show()

class VisualizedTestingWParabola:

    def __init__(self):
        self.real_dim = 2
        self.active_dim = 1

        self.no_samples = 50
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Parameters
        self.sn = 0.1
        self.W = self.kernel.sample_W()

        self.function = Parabola()
        self.real_W = np.asarray([
            [1],
            [1]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

        self.X = np.random.rand(self.no_samples, self.real_dim)
        Z = np.dot(self.X, self.real_W)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        self.w_optimizer = t_WOptimizer(
            self.kernel, # TODO: does the kernel take over the W?
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

        self.no_tries = 1000

    def visualize_quadratic_function(self):
        x_range = np.linspace(0., 1., 80)
        y_range = np.linspace(0., 1., 80)
        X = cartesian([x_range, y_range])

        import os
        if not os.path.exists("./pics/"):
            os.makedirs("./pics/")

        #################################
        #     TRAIN THE W_OPTIMIZER     #
        #################################

        Opt = TripathyOptimizer()

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

            y = self.function._f( np.dot(X, self.real_W).T )
            y_hat = gp_reg.predict(self.X)[0].squeeze()

            #################################
            #   END TRAIN THE W_OPTIMIZER   #
            #################################

            fig = plt.figure()
            ax = Axes3D(fig)

            # First plot the real function
            ax.scatter(X[:,0], X[:, 1], y, s=1)
            ax.scatter(self.X[:,0], self.X[:, 1], y_hat, cmap=plt.cm.jet)
            fig.savefig('./pics/Iter_' + str(j) + '.png', )
            # plt.show()
            plt.close(fig)

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
            np.savetxt("./pics/Iter_" + str(j) + "__" + "Loss_" + str(l) + ".txt", W_hat)

class VisualizedTestingWAugmentedSinusoidal:

    def __init__(self):
        self.real_dim = 2
        self.active_dim = 1

        self.no_samples = 100
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Parameters
        self.sn = 0.1 # 1e-7 # 0.1
        self.W = self.kernel.sample_W()

        self.function = AugmentedSinusoidal()
        self.real_W = np.asarray([
            [3],
            [1]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

        # [[0.9486833]
        #  [0.31622777]]

        x_range = np.linspace(0., 1., int(np.sqrt(self.no_samples)))
        y_range = np.linspace(0., 1., int(np.sqrt(self.no_samples)))
        self.X = cartesian([x_range, y_range])

        #self.X = np.random.rand(self.no_samples, self.real_dim)
        print(self.X.shape)
        Z = np.dot(self.X, self.real_W).reshape(-1, 1)
        print(Z.shape)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        self.w_optimizer = t_WOptimizer(
            self.kernel, # TODO: does the kernel take over the W?
            self.sn,
            np.asscalar(self.kernel.inner_kernel.variance),
            self.kernel.inner_kernel.lengthscale,
            self.X, self.Y
        )

        self.no_tries = 1000
        self.PLOT_MEAN = True

    def visualize_augmented_sinusoidal_function(self):
        x_range = np.linspace(0., 1., 80)
        y_range = np.linspace(0., 1., 80)
        X = cartesian([x_range, y_range])

        import os
        if not os.path.exists("./pics/"):
            os.makedirs("./pics/")

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

            y = self.function._f( np.dot(X, self.real_W).T )

            if self.PLOT_MEAN:
                y_hat = gp_reg.predict(X)[0].squeeze()
            else:
                y_hat = gp_reg.predict(self.X)[0].squeeze()

            #################################
            #   END TRAIN THE W_OPTIMIZER   #
            #################################

            fig = plt.figure()
            ax = Axes3D(fig)

            # First plot the real function
            ax.scatter(X[:,0], X[:, 1], y, s=1)

            if self.PLOT_MEAN:
                ax.scatter(X[:, 0], X[:, 1], y_hat, cmap=plt.cm.jet)
            else:
                ax.scatter(self.X[:,0], self.X[:, 1], y_hat, cmap=plt.cm.jet)


            fig.savefig('./pics/Iter_' + str(j) + '.png', )
            # plt.show()
            plt.close(fig)

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
            np.savetxt("./pics/Iter_" + str(j) + "__" + "Loss_" + str(l) + ".txt", W_hat)

class VisualizeTwoStepOptimization:

    def __init__(self):
        self.real_dim = 2
        self.active_dim = 1

        self.no_samples = 100
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Parameters
        self.sn = 0.1 # 1e-7 # 0.1
        self.W = self.kernel.sample_W()

        self.function = AugmentedSinusoidal()
        self.real_W = np.asarray([
            [3],
            [1]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

        # [[0.9486833]
        #  [0.31622777]]

        x_range = np.linspace(0., 1., int(np.sqrt(self.no_samples)))
        y_range = np.linspace(0., 1., int(np.sqrt(self.no_samples)))
        self.X = cartesian([x_range, y_range])

        #self.X = np.random.rand(self.no_samples, self.real_dim)
        print(self.X.shape)
        Z = np.dot(self.X, self.real_W).reshape(-1, 1)
        print(Z.shape)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        self.optimizer = TripathyOptimizer()

        self.no_tries = 1
        self.PLOT_MEAN = True

    def visualize_augmented_sinusoidal_function(self):
        x_range = np.linspace(0., 1., 80)
        y_range = np.linspace(0., 1., 80)
        X = cartesian([x_range, y_range])

        import os
        if not os.path.exists("./pics-twostep/"):
            os.makedirs("./pics-twostep/")

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

            # TODO: Check if these values are attained over multiple iterations (check if assert otherwise fails)

            # Create the gp_regression function and pass in the predictor function as f_hat
            self.kernel.update_params(W=W_hat, l=l, s=s)
            gp_reg = GPRegression(self.X, self.Y, self.kernel, noise_var=sn)

            y = self.function._f( np.dot(X, self.real_W).T )

            if self.PLOT_MEAN:
                y_hat = gp_reg.predict(X)[0].squeeze()
            else:
                y_hat = gp_reg.predict(self.X)[0].squeeze()

            #################################
            #   END TRAIN THE W_OPTIMIZER   #
            #################################

            fig = plt.figure()
            ax = Axes3D(fig)

            # First plot the real function
            ax.scatter(X[:,0], X[:, 1], y, s=1)

            if self.PLOT_MEAN:
                ax.scatter(X[:, 0], X[:, 1], y_hat, cmap=plt.cm.jet)
            else:
                ax.scatter(self.X[:,0], self.X[:, 1], y_hat, cmap=plt.cm.jet)


            fig.savefig('./pics-twostep/Iter_' + str(j) + '.png', )
            # plt.show()
            plt.close(fig)

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
            np.savetxt("./pics-twostep/Iter_" + str(j) + "__" + "Loss_" + str(l) + ".txt", W_hat)

class VisualizeTwoStepOptimizationWithRestarts:

    def __init__(self):
        self.real_dim = 2
        self.active_dim = 1

        self.no_samples = 100
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        # Parameters
        self.sn = 0.1 # 1e-7 # 0.1
        self.W = self.kernel.sample_W()

        self.function = AugmentedSinusoidal()
        self.real_W = np.asarray([
            [3],
            [1]
        ])
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

        # [[0.9486833]
        #  [0.31622777]]

        x_range = np.linspace(0., 1., int(np.sqrt(self.no_samples)))
        y_range = np.linspace(0., 1., int(np.sqrt(self.no_samples)))
        self.X = cartesian([x_range, y_range])

        #self.X = np.random.rand(self.no_samples, self.real_dim)
        print(self.X.shape)
        Z = np.dot(self.X, self.real_W).reshape(-1, 1)
        print(Z.shape)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        self.optimizer = TripathyOptimizer()

        # self.w_optimizer = t_WOptimizer(
        #     self.kernel, # TODO: does the kernel take over the W?
        #     self.sn,
        #     np.asscalar(self.kernel.inner_kernel.variance),
        #     self.kernel.inner_kernel.lengthscale,
        #     self.X, self.Y
        # )

        self.PLOT_MEAN = True

    def visualize_augmented_sinusoidal_function(self):
        x_range = np.linspace(0., 1., 80)
        y_range = np.linspace(0., 1., 80)
        X = cartesian([x_range, y_range])

        import os
        if not os.path.exists("./pics-twostep/"):
            os.makedirs("./pics-twostep/")

        #################################
        #     TRAIN THE W_OPTIMIZER     #
        #################################

        Opt = TripathyOptimizer()

        print("Real hidden matrix is: ", self.real_W)

        W_hat = self.kernel.sample_W()
        self.kernel.update_params(
            W=W_hat,
            s=self.kernel.inner_kernel.variance,
            l=self.kernel.inner_kernel.lengthscale
        )

        W_hat, sn, l, s = Opt.try_two_step_optimization_with_restarts(self.kernel, self.X, self.Y)

        # TODO: Check if these values are attained over multiple iterations (check if assert otherwise fails)

        # Create the gp_regression function and pass in the predictor function as f_hat
        self.kernel.update_params(W=W_hat, l=l, s=s)
        gp_reg = GPRegression(self.X, self.Y, self.kernel, noise_var=sn)

        y = self.function._f( np.dot(X, self.real_W).T )

        if self.PLOT_MEAN:
            y_hat = gp_reg.predict(X)[0].squeeze()
        else:
            y_hat = gp_reg.predict(self.X)[0].squeeze()

        #################################
        #   END TRAIN THE W_OPTIMIZER   #
        #################################

        fig = plt.figure()
        ax = Axes3D(fig)

        # First plot the real function
        ax.scatter(X[:,0], X[:, 1], y, s=1)

        if self.PLOT_MEAN:
            ax.scatter(X[:, 0], X[:, 1], y_hat, cmap=plt.cm.jet)
        else:
            ax.scatter(self.X[:,0], self.X[:, 1], y_hat, cmap=plt.cm.jet)

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

        fig.savefig('./pics-twostep/BestLoss_' + str(l) + '.png', )
        # plt.show()
        plt.close(fig)

        np.savetxt("./pics-twostep/BestLoss_" + str(l) + ".txt", W_hat)



class TestMatrixRecoveryGreatD:

    def __init__(self):
        self.real_dim = 4
        self.active_dim = 3

        self.no_samples = 25 # This should be proportional to the number of dimensions
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.function = Rosenbrock() # Change the function

        # Generate the positive semi-definite matrix
        self.real_W = np.random.random((self.real_dim, self.active_dim))
        # Take the real W to be completely random
        # assert np.isclose( np.dot(self.real_W.T, self.real_W), np.eye(self.active_dim) )

        self.X = np.random.rand(self.no_samples, self.real_dim)

        #self.X = np.random.rand(self.no_samples, self.real_dim)
        print(self.X.shape)
        Z = np.dot(self.X, self.real_W).reshape(-1, self.active_dim)
        print("The projected input matrix: ", Z.shape)
        print(Z.shape)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        print("Shapes of X and Y: ", (self.X.shape, self.Y.shape) )

        self.optimizer = TripathyOptimizer()

    def check_if_matrix_is_found(self):

        import os
        if not os.path.exists("./highD/"):
            os.makedirs("./highD/")

        #################################
        #     TRAIN THE W_OPTIMIZER     #
        #################################

        Opt = TripathyOptimizer()

        # print("Real hidden matrix is: ", self.real_W)

        W_hat = self.kernel.sample_W()
        self.kernel.update_params(
            W=W_hat,
            s=self.kernel.inner_kernel.variance,
            l=self.kernel.inner_kernel.lengthscale
        )

        W_hat, sn, l, s = Opt.try_two_step_optimization_with_restarts(self.kernel, self.X, self.Y)

        # Create the gp_regression function and pass in the predictor function as f_hat
        self.kernel.update_params(W=W_hat, l=l, s=s)

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

        np.savetxt("./highD/BestLoss_" + str(l) + ".txt", W_hat)
        np.savetxt("./highD/realMatr_" + str(l) + ".txt", self.real_W)

class TestMatrixRecoveryHighDegreePolynomial:

    def __init__(self):
        self.real_dim = 5
        self.active_dim = 1

        self.no_samples = 20 # This should be proportional to the number of dimensions
        self.kernel = TripathyMaternKernel(self.real_dim, self.active_dim)

        self.function = PolynomialKernel() # Change the function

        # Generate the positive semi-definite matrix
        self.real_W = np.random.random((self.real_dim, self.active_dim))
        # Take the real W to be completely random
        # assert np.isclose( np.dot(self.real_W.T, self.real_W), np.eye(self.active_dim) )

        self.X = np.random.rand(self.no_samples, self.real_dim)

        #self.X = np.random.rand(self.no_samples, self.real_dim)
        print(self.X.shape)
        Z = np.dot(self.X, self.real_W).reshape(-1, self.active_dim)
        print("The projected input matrix: ", Z.shape)
        print(Z.shape)
        self.Y = self.function._f(Z.T).reshape(-1, 1)

        print("Shapes of X and Y: ", (self.X.shape, self.Y.shape) )

        self.optimizer = TripathyOptimizer()

    def check_if_matrix_is_found(self):

        import os
        if not os.path.exists("./highD/"):
            os.makedirs("./highD/")

        #################################
        #     TRAIN THE W_OPTIMIZER     #
        #################################

        Opt = TripathyOptimizer()

        # print("Real hidden matrix is: ", self.real_W)

        W_hat = self.kernel.sample_W()
        self.kernel.update_params(
            W=W_hat,
            s=self.kernel.inner_kernel.variance,
            l=self.kernel.inner_kernel.lengthscale
        )

        W_hat, sn, l, s = Opt.try_two_step_optimization_with_restarts(self.kernel, self.X, self.Y)

        # Create the gp_regression function and pass in the predictor function as f_hat
        self.kernel.update_params(W=W_hat, l=l, s=s)

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

        np.savetxt("./highD/" + str(l) + "_BestLoss.txt", W_hat)
        np.savetxt("./highD/" + str(l) + "_realMatr.txt", self.real_W)


if __name__ == "__main__":
    # First, visualize the tau trajectory
    # viz_tau = VisualizedTestingTau()
    # viz_tau.visualize_tau_trajectory_for_random_W()
    # viz_tau.visualize_tau_trajectory_for_identity_W()

    viz_w = TestMatrixRecoveryHighDegreePolynomial()
    viz_w.check_if_matrix_is_found()