from sys import platform

DEV = False
QDev = False
HALFDEV = False

if DEV:

    config = {
        "no_restarts": 2*1, # 20, # 1000

        "max_iter_alg1": 1,  # 1000, # int(1e5),

        "max_iter_parameter_optimization": 1,

        "eps_alg4": 1.e-3,
    }

elif QDev:

    config = {
        # "no_restarts": 16, # 1000
        #
        # "max_iter_alg1": 10, # int(1e5),
        # "max_iter_alg3": 10, # int(1e5),
        #
        # "max_iter_parameter_optimization": 1,
        # "max_iter_W_optimization": 1,
        #
        # "eps_alg1": 1.e-3, # -12,
        # "eps_alg3": 1.e-3, # -12,
        # "eps_alg4": 1.e-3,
        #
        # "tau_max": 1.e-1,
        # "no_taus": 5,
        #
        # "max_dimensions": 3,

        "no_restarts": 12,  # 12, # 1000

        "max_iter_alg1": 50,  # int(1e5),

        "max_iter_parameter_optimization": 30,

        "eps_alg4": 1.e-3,

        "test_single_dimension": True,
    }



elif HALFDEV:

    config = {
        "no_restarts": 25, # 1000

        "max_iter_alg1": 200, # int(1e5),

        "max_iter_parameter_optimization": 100,
        "max_iter_W_optimization": 10,

        "eps_alg4": 8.e-2,

        "test_single_dimension": True,
    }

else:

    config = {
        "no_restarts": 10, # 14,

        "max_iter_alg1": int(300), # # 300 # 100

        "max_iter_parameter_optimization": 200,
        "max_iter_W_optimization": 1,

        "eps_alg4": 1.e-4,

        "test_single_dimension": True,
    }

# Stuff that should be true for all runs:
config['eps_alg1'] = 1.e-4
config['eps_alg3'] = 1.e-6
config['tau_max'] = 1.
config['no_taus'] = 20
config['max_dimensions'] = 2
config['active_dimension'] = 2
config['max_iter_alg3'] = 1
config['std_noise_var'] = 0.005


# In either case, add linux and mac paths
if platform == "linux" or platform == "linux2":
    config['basepath'] = "/home/yedavid/BachelorThesis/bacode/"
    config['dev'] = False
elif platform == "darwin":
    config['basepath'] = "/Users/davidal/GoogleDrive/BachelorThesis/bacode/"
    config['dev'] = True


config['restict_cores'] = True
config['max_cores'] = 16

config['visualize_vanilla_path'] = config['basepath'] + "visualize_vanilla/"
config['visualize_vanilla_vs_gp_path'] = config['basepath'] + "visualize_vanilla_vs_gp/"
config['visualize_angle_loss_path'] = config['basepath'] + "visualize_angle_loss/"

### DATASET SPECIFIC CONFIGURATIONS
config['swissfel_datapath'] = config['basepath'] + "data/swissfel/evaluations.hdf5"
config['projection_datapath'] = config['basepath'] + "data/precomputed_projections/"

config['run_rembo'] = True
config['run_boring'] = True
config['run_tripathy'] = False