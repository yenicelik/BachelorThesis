import sys

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from bacode.tripathy.src.t_kernel import TripathyMaternKernel
from bacode.tripathy.src.t_optimization_functions import t_WOptimizer
from bacode.tripathy.src.t_optimizer import TripathyOptimizer
from febo.environment.benchmarks.functions import Parabola, AugmentedSinusoidal, Rosenbrock, PolynomialKernel
from bacode.tripathy.src.t_loss import loss, dloss_dW, dK_dW, dloss_ds
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


class VisualizeFeatureSelection:

    # F dependent functions
    def setup_environment(self):

        # Environment dimensions
        self.f_input_dim = 2

        # Environment coefficients
        self.a1 = -0.1
        self.a2 = 0.1

        # Setup the projection matrix
        # self.real_W = self.kernel.sample_W()
        self.real_W = np.random.random((self.f_input_dim, self.active_dim))
        self.real_W = self.real_W / np.linalg.norm(self.real_W)

    def f_env(self, x, disp=False):
        real_phi_x = np.concatenate([
            np.square(x[:, 0] - self.a1).reshape(x.shape[0], -1),
            np.square(x[:, 1] - self.a2).reshape(x.shape[0], -1)
        ], axis=1)

        Z = np.dot(real_phi_x, self.real_W).reshape(-1, self.active_dim)
        if not disp:
            assert Z.shape == (self.no_samples, self.active_dim), (Z.shape, (self.no_samples, self.active_dim))
        # print("The projected input matrix: ", Z.shape)
        # print(Z.shape)
        # self.Y = self.function.f(Z.T).reshape(-1, 1)
        out = self.function.f(Z.T).reshape(-1, 1)
        if not disp:
            assert out.shape == (self.no_samples, 1)
        return out

    # G dependent function
    def setup_approximation(self):
        self.g_input_dim = 6
        self.kernel = TripathyMaternKernel(self.g_input_dim, self.active_dim)

    def phi(self, x):
        assert x.shape[1] == 2
        # We could just do x, but at this point it doesn't matter
        phi_x = np.concatenate([
            np.square(x),
            x,
            np.ones_like(x)
        ], axis=1)
        assert phi_x.shape == (self.no_samples, self.g_input_dim)
        return phi_x

    # Part where we generate the data
    def generate_data(self):
        self.X = ( np.random.rand(self.no_samples, self.init_dimension) - 0.5 ) * 2.

        # Generate the real Y
        self.Y = self.f_env(self.X).reshape(-1, 1)
        assert self.Y.shape == (self.no_samples, 1)

        # Generate the kernelized X to use to figure it out
        self.phi_X = self.phi(self.X)

    def __init__(self):
        self.function = Parabola()
        self.init_dimension = 2  # All the input at the beginning is always 1D!
        self.active_dim = self.function.d
        self.no_samples = 30 # 200

        self.setup_environment()
        self.setup_approximation()
        self.generate_data()

        print("Data shape is: ")
        print(self.X.shape)
        print(self.phi_X.shape)

    def plot_3d(self, y_hat, title):

        # Plot real function
        x1 = np.linspace(-1, 1, 100)
        x2 = np.linspace(-1, 1, 100)
        x_real = cartesian([x1, x2])
        y_real = self.f_env(x_real, disp=True)

        # Create the plot
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(azim=30)

        # First plot the real function
        ax.scatter(x_real[:, 0], x_real[:, 1], y_real, 'k.', alpha=.3, s=1)
        ax.scatter(self.X[:, 0], self.X[:, 1], y_hat, cmap=plt.cm.jet)
        fig.savefig('./featureSelection/' + title + '.png')
        plt.show()
        # plt.close(fig)

    def check_if_matrix_is_found(self):

        print("Starting to optimize stuf...")

        import os
        if not os.path.exists("./featureSelection/"):
            os.makedirs("./featureSelection/")

        #################################
        #     TRAIN THE W_OPTIMIZER     #
        #################################

        Opt = TripathyOptimizer()

        print("Real hidden matrix is: ", self.real_W)
        # Start with the approximation of the real matrix

        W_hat = self.kernel.sample_W()
        self.kernel.update_params(
            W=W_hat,
            s=self.kernel.inner_kernel.variance,
            l=self.kernel.inner_kernel.lengthscale
        )

        W_hat, sn, l, s = Opt.try_two_step_optimization_with_restarts(self.kernel, self.phi_X, self.Y)

        # Create the gp_regression function and pass in the predictor function as f_hat
        self.kernel.update_params(W=W_hat, l=l, s=s)
        gp_reg = GPRegression(self.phi_X, self.Y, self.kernel, noise_var=sn)

        # Maybe predict even more values ? (plot the entire surface?)
        y_hat = gp_reg.predict(self.phi_X)[0].squeeze()

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
            self.phi_X,
            self.Y
        )

        np.savetxt("./featureSelection/" + str(l) + "_BestLoss.txt", W_hat)
        np.savetxt("./featureSelection/" + str(l) + "_realMatr.txt", self.real_W)

        # Create the gp_regression function and pass in the predictor function as f_hat
        self.plot_3d(y_hat, title=str(l)+"_BestLoss")


if __name__ == "__main__":
    viz_w = VisualizeFeatureSelection()
    viz_w.check_if_matrix_is_found()
