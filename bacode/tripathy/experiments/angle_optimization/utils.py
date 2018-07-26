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

def pad_2d_list(inp):
    out = []
    m = len(inp)
    n = max([len(x) for x in inp])

    for x in inp:
        diff = n - len(x)
        out.append(x + [x[-1]] * diff )

    # print(out)

    for x in out:
        assert len(x) == n, ("Somehow, not right size!", (m, n), (len(out), len(x)) )
    assert len(out) == m, ("Somehow, not right size!", (m, n), (len(out), len(out[0])) )

    return np.asarray(out)

def calculate_angle_between_two_matrices(A, B):
    M1 = np.dot(A, A.T)
    M2 = np.dot(B, B.T)
    assert M1.shape[0] >= M1.shape[1], ("Not tall matrix!", M1.shape)
    assert M2.shape[0] >= M2.shape[1], ("Not tall matrix!", M2.shape)
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
    # plt.show()

    if not os.path.exists(config['visualize_angle_loss_path']):
        os.makedirs(config['visualize_angle_loss_path'])

    plt.savefig(config['visualize_angle_loss_path'] + title + "_angle" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") )
    plt.clf()

# def visualize_angle_given_W_array(real_projection_A, found_Ws, title):
#     angles = list(map(
#         lambda x: calculate_angle_between_two_matrices(real_projection_A, x),
#         found_Ws
#     ))
#
#     plt.plot(np.arange(len(angles)), angles)
#     plt.ylabel('Angle of found subspace')
#     plt.xlabel('Step of training')
#     plt.show()
#
#     if not os.path.exists(config['visualize_angle_loss_path']):
#         os.makedirs(config['visualize_angle_loss_path'])
#
#     plt.savefig(config['visualize_angle_loss_path'] + title + "_angle" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") )
#     plt.clf()

def visualize_angle_array_stddev(angles, title):
    """

    :param losses: A two dimensional numpy array,
        such that we can calculate the mean and stddev
    :param title:
    :return:
    """
    mean = np.mean(angles, axis=0) # Rowwise mean
    stddev = np.std(angles, axis=0) # Rowwise stddev

    print("Mean shape is: ", mean.shape)
    # assert mean.shape[0] > 3, "Wrong axis!"

    t = np.arange(mean.shape[0])

    # Plot the uncertainty curves
    # plot it!
    fig, ax = plt.subplots(1)
    ax.plot(t, mean, lw=2, label='Difference real', color='blue')
    ax.fill_between(t, mean + stddev, mean - stddev, facecolor='blue', alpha=0.3)
    ax.set_title('Angle between real and found embeddings')
    ax.legend(loc='upper left')
    ax.set_xlabel('Optimization step')
    ax.set_ylabel('Angle between real and found embedding projection')
    ax.grid()

    if not os.path.exists(config['visualize_angle_loss_path']):
        os.makedirs(config['visualize_angle_loss_path'])

    fig.savefig(config['visualize_angle_loss_path'] + "_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_" + title  + "_multiple_angle" )
    plt.clf()

def visualize_loss_array_stddev(loss, title, subtract_mean=False):
    """

    :param losses: A two dimensional numpy array,
        such that we can calculate the mean and stddev
    :param title:
    :return:
    """
    length = loss.shape[1]

    # print("Old loss: ", loss)
    # print("Old loss: ", loss.shape)

    if subtract_mean:
        title = title + "_subtract_mean_"
        loss = loss - np.repeat(np.mean(loss, axis=1).reshape(-1, 1), repeats=length, axis=1)

    # print("New loss: ", loss)
    # print("New loss: ", loss.shape)

    mean = np.mean(loss, axis=0) # Rowwise mean
    stddev = np.std(loss, axis=0) # Rowwise stddev

    print("Mean shape is: ", mean.shape)
    # assert mean.shape[0] > 3, "Wrong axis!"

    t = np.arange(mean.shape[0])

    # Plot the uncertainty curves
    # plot it!
    fig, ax = plt.subplots(1)
    ax.plot(t, mean, lw=2, label='Log likelihood loss found parameters', color='blue')
    ax.fill_between(t, mean + stddev, mean - stddev, facecolor='blue', alpha=0.3)
    ax.set_title('Log likelihood loss of the Gaussian Process')
    ax.legend(loc='upper left')
    ax.set_xlabel('Optimization step')
    ax.set_ylabel('Log likelihood loss of the Gaussian Process of the found parameters')
    ax.grid()

    if not os.path.exists(config['visualize_angle_loss_path']):
        os.makedirs(config['visualize_angle_loss_path'])

    fig.savefig(config['visualize_angle_loss_path'] + "_" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_" + title + "_multiple_loss" )
    plt.clf()