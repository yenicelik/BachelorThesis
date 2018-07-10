import sys

from bacode.tripathy.src.bilionis_refactor.config import config

sys.path.append("/Users/davidal/GoogleDrive/BachelorThesis/bacode/tripathy")
print(sys.path)
import numpy as np
# from bacode.tripathy.src.bilionis.t_kernel import TripathyMaternKernel
# from bacode.tripathy.src.bilionis.t_optimizer import TripathyOptimizer
# from febo.environment.benchmarks.functions import Parabola
# from bacode.tripathy.src.bilionis.t_loss import loss

from bacode.tripathy.experiments.angle_optimization.utils import calculate_angle_between_two_matrices

from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel
from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer
from febo.environment.benchmarks.functions import Parabola
from bacode.tripathy.src.bilionis_refactor.t_loss import loss

"""
    ParallelLbfgs
"""

from febo.utils.utils import cartesian

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
        # self.real_W = np.random.random((self.f_input_dim, self.active_dim))
        # self.real_W = self.real_W / np.linalg.norm(self.real_W)
        self.real_W = np.asarray([
            [5.889490030086947936e-01,
             8.081701998063678394e-01
             ]
        ]).T

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
        self.g_input_dim = 5
        self.kernel = TripathyMaternKernel(self.g_input_dim, self.active_dim)

    def phi(self, x):
        assert x.shape[1] == 2
        # We could just do x, but at this point it doesn't matter
        phi_x = np.concatenate([
            np.square(x),
            x,
            np.ones((x.shape[0],)).reshape(-1, 1)
        ], axis=1)
        # print("Phi x is: ", phi_x)
        assert phi_x.shape == (self.no_samples, self.g_input_dim)
        return phi_x

    # Part where we generate the data
    def generate_data(self):
        self.X = (np.random.rand(self.no_samples, self.init_dimension) - 0.5) * 2.

        # Generate the real Y
        self.Y = self.f_env(self.X).reshape(-1, 1)
        assert self.Y.shape == (self.no_samples, 1)

        # Generate the kernelized X to use to figure it out
        self.phi_X = self.phi(self.X)

    def __init__(self):
        self.function = Parabola()
        self.init_dimension = 2  # All the input at the beginning is always 1D!
        self.active_dim = self.function.d
        self.no_samples = 50  # 200

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

        np.savetxt(config['basepath'] + "/featureSelection/" + str(l) + "_BestLoss.txt", W_hat)
        np.savetxt(config['basepath'] + "/featureSelection/" + str(l) + "_realMatr.txt", self.real_W)

        # Create the gp_regression function and pass in the predictor function as f_hat
        self.plot_3d(y_hat, title=str(l) + "_BestLoss")


if __name__ == "__main__":
    viz_w = VisualizeFeatureSelection()
    viz_w.check_if_matrix_is_found()

    realW = np.asarray([[
        0.589,
        0.808,
        0.118,
        -0.162,
        0.823
    ]])
    # 
    # # dump all matrices:
    all_arrs = [np.asarray([[-0.53231756],
                [-0.58606002],
                [-0.08567194],
                [0.23366606],
                [-0.55788185]]), np.asarray([[-0.51911108],
       [-0.73661668],
       [-0.18156216],
       [ 0.11700797],
       [ 0.37585087]]), np.asarray([[ 0.55927232],
       [ 0.62212119],
       [ 0.4304515 ],
       [-0.33877583],
       [-0.01105196]]), np.asarray([[-0.0700867 ],
       [ 0.44039708],
       [ 0.00547101],
       [ 0.87285038],
       [ 0.1980923 ]]), np.asarray([[-0.20621242],
       [-0.87978725],
       [-0.01425874],
       [ 0.22520717],
       [-0.36404568]]), np.asarray([[ 0.44436629],
       [-0.27324693],
       [-0.35247927],
       [-0.42161812],
       [ 0.65258811]]), np.asarray([[ 0.02757661],
       [ 0.03507293],
       [ 0.22857683],
       [ 0.95362674],
       [ 0.19067796]]), np.asarray([[-0.17595457],
       [ 0.46358691],
       [-0.18580369],
       [-0.84829019],
       [-0.00281296]]), np.asarray([[-0.4968206 ],
       [-0.54711683],
       [-0.19065551],
       [ 0.14520442],
       [ 0.62960196]]), np.asarray([[ 0.2653316 ],
       [-0.20752778],
       [ 0.40746608],
       [ 0.06160371],
       [ 0.84658593]]), np.asarray([[ 0.11290238],
       [-0.26906469],
       [ 0.32163767],
       [ 0.9004653 ],
       [-0.02384746]]), np.asarray([[-0.14686325],
       [ 0.33728604],
       [-0.00712895],
       [-0.69460021],
       [ 0.61818205]]), np.asarray([[-0.16968701],
       [ 0.01701022],
       [-0.01233551],
       [ 0.71393683],
       [-0.67901326]]), np.asarray([[-0.35545982],
       [-0.53271436],
       [-0.09082412],
       [ 0.09902547],
       [-0.75618031]]), np.asarray([[-0.72894047],
       [-0.19448231],
       [ 0.58556318],
       [ 0.05825711],
       [-0.29076501]]), np.asarray([[-0.19770904],
       [ 0.15302142],
       [-0.30356534],
       [-0.04928292],
       [ 0.91810395]]), np.asarray([[ 0.18858497],
       [-0.18336775],
       [ 0.10141913],
       [ 0.95910771],
       [ 0.02526946]]), np.asarray([[-0.31897263],
       [-0.66603309],
       [-0.04958786],
       [ 0.30999695],
       [-0.59674058]]), np.asarray([[ 0.28386614],
       [ 0.34085778],
       [ 0.1031528 ],
       [-0.04421027],
       [ 0.88917993]]), np.asarray([[ 0.16444762],
       [-0.41779061],
       [ 0.0107646 ],
       [ 0.71935132],
       [ 0.52992998]]), np.asarray([[-0.47542077],
       [-0.77290449],
       [-0.36145487],
       [-0.1569422 ],
       [ 0.14599062]]), np.asarray([[-0.49707585],
       [-0.71544881],
       [-0.12881437],
       [ 0.12912382],
       [ 0.45583164]]), np.asarray([[-0.40424927],
       [-0.60047642],
       [-0.06233213],
       [ 0.14459672],
       [-0.67172695]]), np.asarray([[-0.3571697 ],
       [-0.24498347],
       [ 0.05157643],
       [ 0.16184898],
       [ 0.88518794]]), np.asarray([[-0.72676113],
       [-0.60188446],
       [-0.20705528],
       [ 0.17194289],
       [-0.19265802]])]

    goodFoundMatrix = np.asarray([[
        -3.554598211439893851e-01,
        - 5.327143597510963779e-01,
        - 9.082412386372847035e-02,
        9.902546601936182413e-02,
        - 7.561803105551594406e-01
    ]])

    angle = calculate_angle_between_two_matrices(goodFoundMatrix.T, realW.T)
    # print(angle)

    # best_W = None
    # best_angle = 1000
    #
    # for arr in all_arrs:
    #     # print("Matrix is: ", arr)
    #     # new_arr = np.asarray([[
    #     #     arr[0, 0],
    #     #     arr[1, 0],
    #     #     arr[2, 0],
    #     #     arr[3, 0],
    #     #     arr[4, 0] + arr[5, 0]
    #     # ]]).T
    #     # print("New matrix is: ", new_arr)
    #     angle = calculate_angle_between_two_matrices(realW.T, arr)
    #     print(angle)
    #     if np.abs(angle) < best_angle:
    #         best_angle = angle
    #         best_W = arr
    #
    # print("Best values are: ", best_angle, best_W)
