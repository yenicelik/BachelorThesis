import sys
sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/code/tripathy")
print(sys.path)
import numpy as np
from src.t_kernel import TripathyMaternKernel
from src.t_optimization_functions import t_WOptimizer
from src.t_optimizer import TripathyOptimizer
from febo.environment.benchmarks.functions import Parabola
from src.t_loss import loss, dloss_dW, dK_dW, dloss_ds
import matplotlib.pyplot as plt

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

class VisualizedTestingW:

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


if __name__ == "__main__":
    # First, visualize the tau trajectory
    viz_tau = VisualizedTestingTau()
    # viz_tau.visualize_tau_trajectory_for_random_W()
    # viz_tau.visualize_tau_trajectory_for_identity_W()

    viz_w = VisualizedTestingW()
    viz_w.visualize_quadratic_function()