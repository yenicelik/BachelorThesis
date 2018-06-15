"""
    Check if the function can be succesfully approximated using the GP
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from febo.models.gpy import GPRegression
from febo.utils.utils import cartesian

from bacode.tripathy.src.bilionis_refactor.t_optimizer import TripathyOptimizer
from bacode.tripathy.src.bilionis_refactor.t_kernel import TripathyMaternKernel

from bacode.tripathy.src.test_real_data.swissfel.importer import import_X_Y
from bacode.tripathy.src.test_real_data.metrics import l2difference

default_tuple = ()

def run_optimization(X, Y, Y_noise, input_d):

    # TODO: Y_noise is notuuse!
    assert X.shape[1] == input_d, (X.shape, input_d)

    # Run the optimizer
    optimizer = TripathyOptimizer()
    W_hat, sn, l, s, d = optimizer.find_active_subspace(X, Y)

    # Spawn new kernel
    kernel = TripathyMaternKernel(
        real_dim=input_d,
        active_dim=d,
        W=W_hat,
        variance=s,
        lengthscale=l
    )

    # Spawn new GPRegression object
    gpreg = GPRegression(
        input_dim=input_d,
        kernel=kernel,
        noise_var=sn,
        calculate_gradients=True
    )

    # Return the optimized gpreg object
    return gpreg

def make_prediction(gpreg, X):
    return gpreg.predict(Xnew=X)

def get_visualizable_X(X, W):
    assert X.shape[1] == W.shape[0]
    U, E, V = np.linalg.svd(W)
    U = U[:,:2]
    print(U.shape)
    return np.dot(X, U)

def run_main():
    no_measures = 5
    X, Y, noise = import_X_Y()

    # Add one point after the other, and check the loss (how the loss enhances etc.)
    assert Y.shape == (X.shape[0],), (Y.shape, X.shape)

    gp_reg = run_optimization(X=X[:, :], Y=Y[:], Y_noise=noise, input_d=X.shape[1])
    y_hat = make_prediction(gp_reg, X[:, :])
    loss = l2difference(y_hat, Y[:])
    likelihood = gp_reg.log_likelihood()

    print("Loss is: ", loss)
    print("Likelihood is: ", likelihood)
    # We can also print out the log-likelihood:

    # Create the plot
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.view_init(azim=30)

    # First plot the real function
    print("Generated W is the following: ", gp_reg.kern.W)
    title = str(loss) + "_save"
    import os
    os.makedirs("./swissfelPred/", exist_ok=True)
    np.savetxt("./swissfelPred/" + title + "_realMatr.txt", gp_reg.kern.W)

    # Concatenate X by grid values
    # max_X_val = np.min(X)
    # min_X_val = np.min(X)
    # step = int( (max_X_val - min_X_val) // 10. )
    #
    # real_VIZ_X = get_visualizable_X(X, gp_reg.kern.W)
    #
    # # Visualize the prediction of the GP
    # X_grid = cartesian(
    #     [
    #         np.arange(np.min(X[:,i]), np.max(X[:,i]), step, dtype=int) for i in range(X.shape[1])
    #     ]
    # )
    # y_pred = gp_reg.predict(X_grid)
    # VIZ_X = get_visualizable_X(X_grid, gp_reg.kern.W)

    VIZ_X = get_visualizable_X(X, gp_reg.kern.W)

    ax = plt.axes(projection='3d')
    ax.plot_trisurf(VIZ_X[:, 0], VIZ_X[:, 1], Y, cmap='viridis', edgecolor='none', alpha=.3)

    # # Create the plot
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.view_init(azim=30)
    #
    # # First plot the real function
    # ax.scatter(VIZ_X[:, 0], VIZ_X[:, 1], y_pred, 'k.', alpha=.3, s=1)
    # ax.scatter(real_VIZ_X[:, 0], real_VIZ_X[:, 1], Y, cmap=plt.cm.jet)
    #
    # fig.savefig('./swissfelPred/' + title + '.png')
    # plt.show()

    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(VIZ_X[:, 0], VIZ_X[:, 1], Y, cmap='viridis', edgecolor='none', alpha=.3)

    # Visualize losses
    plt.savefig("./swissfelPred/" + title + "_vizProjection.png")
    plt.show()

if __name__ == "__main__":
    print("Checkin how well our model predicts the swissfel dataset")
    run_main()
