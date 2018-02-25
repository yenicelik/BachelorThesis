
class AcquisitionFunction:

    def __init__(self):
        pass

    def ucb(x, gp, beta):
        mean, stddev = gp.predict_single(x)
        return mean + beta * stddev


class Optimizer:

    def __init__(self):
        pass

    def acq_max(self, acq_fnc, gp, y_max, bounds, random_state, n_warmup=100000, n_iter=250):
        """
            A function to find the maximum of the acquisition function
            It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
            optimization method. First by sampling `n_warmup` (1e5) points at random,
            and then running L-BFGS-B from `n_iter` (250) random starting points.
            Parameters
            ----------
            :param ac:
                The acquisition function object that return its point-wise value.
            :param gp:
                A gaussian process fitted to the relevant data.
            :param y_max:
                The current maximum known value of the target function.
            :param bounds:
                The variables bounds to limit the search of the acq max.
            :param random_state:
                instance of np.RandomState random number generator
            :param n_warmup:
                number of times to randomly sample the aquisition function
            :param n_iter:
                number of times to run scipy.minimize
            Returns
            -------
            :return: x_max, The arg max of the acquisition function.
        """
        # Warm up with random points
        x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warmup, bounds.shape[0]))

        ys = acq_fnc(x_tries, gp=gp, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()

        # Explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0]))
        
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -acq_fnc(x.reshape(1, -1), gp=gp, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")

            # See if success
            if not res.success:
                continue
               
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])
