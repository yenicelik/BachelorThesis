"""
    All common functions that are shared between the different visualization modules
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_angle_between_two_matrices(A, B):
    M1 = np.dot(A, A.T)
    M2 = np.dot(B, B.T)
    diff = np.linalg.norm(M1 - M2, ord=2)
    return np.arcsin(diff)

def visualize_angle_given_W_array(real_projection_A, found_Ws):
    angles = list(map(
        lambda x: calculate_angle_between_two_matrices(real_projection_A, x),
        found_Ws
    ))

    plt.plot(np.arange(len(angles)), angles)
    plt.ylabel('Angle of found subspace')
    plt.xlabel('Step of training')
    plt.show()

    pass


def spawn_training_points():
    pass