import numpy as np
from numpy import AxisError
from scipy.stats import zscore
from .data_import import *
import pprint
# All functions here can expect to handle the output from BCMD Model i.e.
# a dict.


class Error(Exception):
    """
    Base class for exceptions in this module.
    """
    pass


class ZeroArrayError(Error):
    """
    Exception raised for errors in the zero array.
    """
    pass


def euclidean_dist(data1, data2):
    """
    Gives the euclidean distance between two numpy arrays.

    :param data1: Numpy array for data1
    :type data1: np.ndarray
    :param data2: Numpy array for data2
    :type data2: np.ndarray

    :return: Euclidean distance measure
    :rtype: list of float
    """
    try:
        assert(data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
    try:
        d = np.sum(np.sqrt(np.sum((data1 - data2) * (data1 - data2), axis=1)))
    except AxisError:
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        return None
    return d


def manhattan_dist(data1, data2):
    """
    Gives the Manhattan distance between two numpy arrays.

    :param data1: Numpy array for data1
    :type data1: np.ndarray
    :param data2: Numpy array for data2
    :type data2: np.ndarray

    :return: Manhattan distance measure
    :rtype: list of float
    """
    assert(data1.shape == data2.shape), 'Arrays not of equal size'
    return np.sum(np.abs(data1 - data2))


def mean_square_error_dist(data1, data2):
    """
    Gives the mean squared error between two numpy arrays.

    :param data1: Numpy array for data1
    :type data1: np.ndarray
    :param data2: Numpy array for data2
    :type data2: np.ndarray

    :return: MSE distance measure
    :rtype: list of float
    """
    assert(data1.shape == data2.shape), 'Arrays not of equal size'
    n = data1.shape[1]
    return np.sum(1 / n * np.sum((data1 - data2) * (data1 - data2), axis=1))


def mean_absolute_error_dist(data1, data2):
    """
    Gives the normalised manhattan distance between two numpy arrays.

    :param data1: Numpy array for data1
    :type data1: np.ndarray
    :param data2: Numpy array for data2
    :type data2: np.ndarray

    :return: MAE distance measure
    :rtype: list of float
    """
    assert(data1.shape == data2.shape), 'Arrays not of equal size'
    n = data1.shape[1]
    return 1 / n * np.sum(np.abs(data1 - data2))


DISTANCES = {
    'euclidean': euclidean_dist,
    'manhattan': manhattan_dist,
    'MSE': mean_square_error_dist,
    'MAE': mean_absolute_error_dist
}


def check_for_key(dictionary, target):
    try:
        data = dictionary[target]
    except KeyError:
        print('Actual data does not contain target value.')
    return data


def zero_array(array, zero_flags):
    """
    Method to zero an array of data with the initial values.
    :param array: Array of data - rows are signals, columns are timepoints.
    :return: Zero'd numpy array
    :rtype: np.ndarray
    """
    init = array[:, 0] * zero_flags
    zerod = np.apply_along_axis(lambda x: x - init, 0, array)
    return zerod


def get_distance(actual_data, sim_data, targets,
                 distance='euclidean', zero_flag=None,
                 normalise=False):

    d0 = []
    d_star = []
    for idx, k in enumerate(targets):
        d0.append(check_for_key(actual_data, k))
        d_star.append(check_for_key(sim_data, k))
    d0 = np.array(d0)
    d_star = np.array(d_star)

    if zero_flag is not None:
        if len(zero_flag) == len(targets):
            try:
                d_star = zero_array(np.array(d_star), zero_flag)
            except (TypeError, IndexError):
                print('Invalid Data', end="\r")
                return (float('NaN'))
        else:
            raise ZeroArrayError("Length of zero array didn't match targets")

    if normalise:
        try:
            norm_d0 = zscore(d0, axis=1)
            norm_d_star = zscore(d_star, axis=1)
        except TypeError as e:
            print(d_star)
            raise e
        distances = {"TOTAL": DISTANCES[distance](norm_d0, norm_d_star)}
    else:
        distances = {"TOTAL": DISTANCES[distance](d0, d_star)}

    for k in targets:
        d1 = check_for_key(actual_data, k)
        d1 = np.array(d1).reshape(1, len(d1))

        d2 = check_for_key(sim_data, k)
        d2 = np.array(d2).reshape(1, len(d2))

        if normalise:
            try:
                d1 = zscore(d1, axis=1)
            except TypeError as e:
                print(d1)
                raise e
            try:
                d2 = zscore(d2, axis=1)
            except TypeError as e:
                print(d2)
                raise e

        distances[k] = DISTANCES[distance](d1, d2)

    return distances
