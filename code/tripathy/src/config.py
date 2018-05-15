from sys import platform

DEV = False
HALFDEV = False

if DEV:

    config = {
        "no_restarts": 4*10, # 20, # 1000

        "max_iter_alg1": 1,  # 1000, # int(1e5),
        "max_iter_alg3": 1, # 1000, # int(1e5),

        "max_iter_parameter_optimization": 1,
        "max_iter_W_optimization": 1,

        "eps_alg1": 1.e-3, # -12,
        "eps_alg3": 1.e-3, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 10,
    }

elif HALFDEV:

    config = {
        "no_restarts": 16*50, # 1000

        "max_iter_alg1": 1000, # int(1e5),
        "max_iter_alg3": 1000, # int(1e5),

        "max_iter_parameter_optimization": 2,
        "max_iter_W_optimization": 2,

        "eps_alg1": 1.e-6, # -12,
        "eps_alg3": 1.e-6, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 10,
    }

else:

    config = {
        "no_restarts": 16*100,

        "max_iter_alg1": int(1e5),
        "max_iter_alg3": int(1e5),

        "max_iter_parameter_optimization": 1,
        "max_iter_W_optimization": 1,

        "eps_alg1": 1.e-9, # -12,
        "eps_alg3": 1.e-9, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 100,
    }

# In either case, add linux and mac paths
if platform == "linux" or platform == "linux2":
    config['basepath'] = "/home/yedavid/BachelorThesis/"
    config['dev'] = False
elif platform == "darwin":
    config['basepath'] = "/Users/davidal/GoogleDrive/BachelorThesis/code"
    config['dev'] = True