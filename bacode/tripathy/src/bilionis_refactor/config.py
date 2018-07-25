from sys import platform

DEV = True
QDev = False
HALFDEV = False

if DEV:

    config = {
        "no_restarts": 2*1, # 20, # 1000

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
        "max_iter_alg3": 50,  # int(1e5),

        "max_iter_parameter_optimization": 30,
        "max_iter_W_optimization": 1,

        "eps_alg1": 1.e-4,  # -12,
        "eps_alg3": 1.e-4,  # -12,
        "eps_alg4": 1.e-3,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "test_single_dimension": True,
        "max_dimensions": 2,
    }



elif HALFDEV:

    config = {
        "no_restarts": 25, # 1000

        "max_iter_alg1": 200, # int(1e5),
        "max_iter_alg3": 200, # int(1e5),

        "max_iter_parameter_optimization": 100,
        "max_iter_W_optimization": 10,

        "eps_alg1": 1.e-12, # -12,
        "eps_alg3": 1.e-12, # -12,
        "eps_alg4": 8.e-2,

        "tau_max": 1.e-1,
        "no_taus": 6,

        "test_single_dimension": True,
        "max_dimensions": 3,
    }

else:

    config = {
        "no_restarts": 24, # 14,

        "max_iter_alg1": int(400), # 100
        "max_iter_alg3": int(2000),

        "max_iter_parameter_optimization": 200,
        "max_iter_W_optimization": 1,

        "eps_alg1": 5.e-12, # -12,
        "eps_alg3": 5.e-12, # -12,
        "eps_alg4": 1.e-4,

        "tau_max": 1.e-1,
        "no_taus": 5,

        "test_single_dimension": True,

        "max_dimensions": 100,
    }

# In either case, add linux and mac paths
if platform == "linux" or platform == "linux2":
    config['basepath'] = "/home/yedavid/BachelorThesis/bacode/"
    config['dev'] = False
elif platform == "darwin":
    config['basepath'] = "/Users/davidal/GoogleDrive/BachelorThesis/bacode/"
    config['dev'] = True

config['active_dimension'] = 2

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