DEV = True
HALFDEV = False

if DEV:

    config = {
        "no_restarts": 1, # 20, # 1000

        "max_iter_alg1": 1,  # 1000, # int(1e5),
        "max_iter_alg3": 1, # 1000, # int(1e5),

        "max_iter_parameter_optimization": 10,
        "max_iter_W_optimization": 20,

        "eps_alg1": 1.e-3, # -12,
        "eps_alg3": 1.e-3, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 10,
    }

elif HALFDEV:

    config = {
        "no_restarts": 20, # 1000

        "max_iter_alg1": 1000, # int(1e5),
        "max_iter_alg3": 1000, # int(1e5),

        "max_iter_parameter_optimization": 50,
        "max_iter_W_optimization": 50,

        "eps_alg1": 1.e-3, # -12,
        "eps_alg3": 1.e-3, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 10,
    }

else:

    config = {
        "no_restarts": 1000,

        "max_iter_alg1": int(1e5),
        "max_iter_alg3": int(1e5),

        "max_iter_parameter_optimization": 50,
        "max_iter_W_optimization": 50,

        "eps_alg1": 1.e-3, # -12,
        "eps_alg3": 1.e-3, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 10,
    }