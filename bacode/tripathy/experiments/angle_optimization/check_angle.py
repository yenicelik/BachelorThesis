import numpy as np

from utils import calculate_angle_between_two_matrices

parabola_embedding = np.asarray([[0.49969147, 0.1939272]]).T

camelback_embedding = np.asarray([
    [-0.31894555, 0.78400512, 0.38970008, 0.06119476, 0.35776912],
    [-0.27150973, 0.066002, 0.42761931, -0.32079484, -0.79759551]
]).T

sinusoidal_embedding = np.asarray([
    [-0.41108301, 0.22853536, -0.51593653, -0.07373475, 0.71214818],
    [0.00412458, -0.95147725, -0.28612815, -0.06316891, 0.093885]
]).T

if __name__ == "__main__":
    print("Calculating angle between the two matrices")

    # comparison_matrix = np.asarray([[0.42534926, 0.10170223],
    #                                 [0.22599586, -0.96462682],
    #                                 [0.51772184, 0.01648976],
    #                                 [0.07458378, 0.08468852],
    #                                 [-0.70313955, -0.22739327]
    #                                 ])
    comparison_matrix = np.asarray([
        [0.92752153],
        [0.37376973]
    ])

    if True:
        print("Angle between the real parabola and the found matrices are: ")
        print(calculate_angle_between_two_matrices(parabola_embedding, comparison_matrix))
    elif False:
        print("Angle between the real camelback and the found matrices are: ")
        print(calculate_angle_between_two_matrices(camelback_embedding, comparison_matrix))
    else:
        print("Angle between the real sinusoidal and the found matrices are: ")
        print(calculate_angle_between_two_matrices(sinusoidal_embedding, comparison_matrix))
