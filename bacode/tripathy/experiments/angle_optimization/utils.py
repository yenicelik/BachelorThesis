"""
    All common functions that are shared between the different visualization modules
"""
import datetime

import numpy as np
import matplotlib
import os

from bacode.tripathy.src.bilionis_refactor.config import config

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calculate_angle_between_two_matrices(A, B):
    M1 = np.dot(A, A.T)
    M2 = np.dot(B, B.T)
    assert M1.shape == M2.shape, ("Not same shapes!", M1.shape, M2.shape)
    assert M1.shape[0] > 1
    diff = np.linalg.norm(M1 - M2, ord=2)
    print("Diff is: ", diff)
    out = np.arcsin(diff)
    out = np.rad2deg(out)
    print("Out is: ", out)

    return out

def visualize_angle_given_W_array(real_projection_A, found_Ws, title):
    angles = list(map(
        lambda x: calculate_angle_between_two_matrices(real_projection_A, x),
        found_Ws
    ))

    plt.plot(np.arange(len(angles)), angles)
    plt.ylabel('Angle of found subspace')
    plt.xlabel('Step of training')
    plt.show()

    if not os.path.exists(config['visualize_angle_loss_path']):
        os.makedirs(config['visualize_angle_loss_path'])

    plt.savefig(config['visualize_angle_loss_path'] + title + "_angle" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") )
    plt.clf()

def visualize_loss_given_loss_array(losses, title):
    pass