from sys import platform

DEV = False
QDev = True
HALFDEV = True

if DEV:

    config = {
        "no_restarts": 4*1, # 20, # 1000

        "max_iter_alg1": 1,  # 1000, # int(1e5),
        "max_iter_alg3": 1, # 1000, # int(1e5),

        "max_iter_parameter_optimization": 1,
        "max_iter_W_optimization": 1,

        "eps_alg1": 1.e-2, # -12,
        "eps_alg3": 1.e-2, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 3,
    }

elif QDev:

    config = {
        "no_restarts": 12, # 1000

        "max_iter_alg1": 30, # int(1e5),
        "max_iter_alg3": 30, # int(1e5),

        "max_iter_parameter_optimization": 1,
        "max_iter_W_optimization": 1,

        "eps_alg1": 1.e-3, # -12,
        "eps_alg3": 1.e-3, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 2,
    }

elif HALFDEV:

    config = {
        "no_restarts": 4*4*16, # 1000

        "max_iter_alg1": 400, # int(1e5),
        "max_iter_alg3": 400, # int(1e5),

        "max_iter_parameter_optimization": 1,
        "max_iter_W_optimization": 1,

        "eps_alg1": 1.e-4, # -12,
        "eps_alg3": 1.e-4, # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "max_dimensions": 2,
    }

else:

    config = {
        "no_restarts": 16*15,

        "max_iter_alg1": int(4000),
        "max_iter_alg3": int(4000),

        "max_iter_parameter_optimization": 1,
        "max_iter_W_optimization": 1,

        "eps_alg1": 1.e-4, # -12,
        "eps_alg3": 1.e-4, # -12,
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
    config['basepath'] = "/Users/davidal/GoogleDrive/BachelorThesis/bacode/"
    config['dev'] = True

config['restict_cores'] = True
config['max_cores'] = 16


### DATASET SPECIFIC CONFIGURATIONS
config['swissfel_datapath'] = config['basepath'] + "data/swissfel/evaluations.hdf5"