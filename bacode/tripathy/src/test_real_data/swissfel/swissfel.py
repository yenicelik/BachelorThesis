"""
    Check if the function can be succesfully approximated using the GP
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from febo.models.gpy import GPRegression

import bacode.tripathy.src
from bacode.tripathy.src.t_optimizer import TripathyOptimizer
from bacode.tripathy.src.t_kernel import TripathyMaternKernel

from bacode.tripathy.src.test_real_data.swissfel.importer import import_X_Y
from bacode.tripathy.src.test_real_data.metrics import l2difference

default_tuple = ()

def run_optimization(X, Y, Y_noise, input_d):

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
    step = X.shape[0]//no_measures
    max_train_proportion = int(X.shape[0] * 0.9)
    losses = []
    points = []
    loglikelihoods = []
    for i in range(step, max_train_proportion, step):
        print("Using first " + str(i) + " as training data, and " + str(X.shape[0] - i) + " as eval data")

        assert Y.shape == (X.shape[0],), (Y.shape, X.shape)

        gp_reg = run_optimization(X=X[:i, :], Y=Y[:i], Y_noise=noise, input_d=X.shape[1])
        y_hat = make_prediction(gp_reg, X[i:, :])
        loss = l2difference(y_hat, Y[i:])
        loglikelihoods.append(gp_reg.log_likelihood())

        print("Loss is: ", loss)


        # We can also print out the log-likelihood:

        losses.append(loss)
        points.append(i)

    print(losses)

    # Create the plot
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.view_init(azim=30)

    # First plot the real function
    VIZ_X = get_visualizable_X(X, gp_reg.kern.W)

    ax = plt.axes(projection='3d')
    ax.plot_trisurf(VIZ_X[:, 0], VIZ_X[:, 1], Y, cmap='viridis', edgecolor='none', alpha=.3)

    fig2 = plt.figure(2)
    # Visualize losses
    plt.plot(points, losses, label="loss")
    plt.plot(points, loglikelihoods, label="likelihood")
    plt.ylabel('Loss over datapoints')
    plt.legend(loc=2)
    plt.show()

if __name__ == "__main__":
    print("Checkin how well our model predicts the swissfel dataset")
    run_main()
