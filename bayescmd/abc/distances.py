# -*- coding: utf-8 -*-
"""Use to generate distance measures between simulated and real time series.

Attributes
----------
DISTANCES : dict
    Dictionary containing the distance aliases, mapping to the functions.

"""
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Import DTW distance functions
from dtaidistance import dtw
# Comment this line out as it appears to be deprecated.
# from numpy import AxisError

# All functions here can expect to handle the output from BCMD Model i.e.
# a dict.


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class ZeroArrayError(Error):
    """Exception raised for errors in the zero array."""

    pass


def dtw_distance(data1, data2):
    """Get the DTW distance between two numpy arrays.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    Returns
    -------
    d : float
        DTW distance measure
    """

    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e

    try:
        data1.shape[1]
    except IndexError:
        print("Reshaping data1 to have 2 dimensions")
        data1 = data1.reshape((-1, 1))

    try:
        data2.shape[1]
    except IndexError:
        print("Reshaping data2 to have 2 dimensions")
        data2 = data2.reshape((-1, 1))

    try:
        d = 0
        for ii in range(len(data1)):
            d += dtw.distance_fast(np.array(data1[ii], dtype=np.double),
                                   np.array(data2[ii], dtype=np.double))
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e

    except TypeError as e:
        print("Null output - distance set to None")
        d = None
    return d


def dtw_weighted_distance(data1, data2):
    """Get the weighted DTW distance between two numpy arrays.

    Data 2 assumed to be real/true data.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    Returns
    -------
    d : float
        DTW distance measure
    """

    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e

    try:
        data1.shape[1]
    except IndexError:
        print("Reshaping data1 to have 2 dimensions")
        data1 = data1.reshape((-1, 1))

    try:
        data2.shape[1]
    except IndexError:
        print("Reshaping data2 to have 2 dimensions")
        data2 = data2.reshape((-1, 1))

    try:
        d = 0
        for ii in range(len(data1)):
            rng = np.max(data2[ii]) - np.min(data2[ii])
            d += dtw.distance_fast(np.array(data1[ii], dtype=np.double),
                                   np.array(data2[ii], dtype=np.double))/rng
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    except TypeError as e:
        print("Null output - distance sset to None")
        d = None
    return d


def euclidean_dist(data1, data2):
    """Get the euclidean distance between two numpy arrays.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    Returns
    -------
    d : float
        Euclidean distance measure

    """
    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    try:
        d = np.sum(np.sqrt(np.sum((data1 - data2) * (data1 - data2), axis=1)))
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e

    return d


def manhattan_dist(data1, data2):
    """Get the Manhattan distance between two numpy arrays.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    Returns
    -------
    d : float
        Manhattan distance measure

    """
    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e

    try:
        d = np.sum(np.abs(data1 - data2))
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    return d


def mean_square_error_dist(data1, data2):
    """Get the Mean Square Error distance between two numpy arrays.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    Returns
    -------
    d : float
        Mean Square Error distance measure

    """
    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    assert (data1.shape == data2.shape), 'Arrays not of equal size'

    # Get number of time points to average over.
    n = data1.shape[1]
    try:
        d = np.sum(1 / n * np.sum((data1 - data2) * (data1 - data2), axis=1))
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    return d


def root_mean_square_error_dist(data1, data2):
    """Get the Root Mean Square Error distance between two numpy arrays.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    Returns
    -------
    d : float
        Root Mean Square Error distance measure

    """
    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    assert (data1.shape == data2.shape), 'Arrays not of equal size'

    # Get number of time points to average over.
    n = data1.shape[1]
    try:
        d = np.sum(np.sqrt(1 / n * np.sum((data1 - data2) * (data1 - data2),
                                          axis=1)))
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    return d


def normalised_root_mean_square_error_dist(data1, data2):
    """Get the Normalised Root Mean Square Error distance between two numpy
    arrays.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows. Should generally be the measured data i.e. d0.

    Returns
    -------
    d : float
    """
    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    assert (data1.shape == data2.shape), 'Arrays not of equal size'

    # Get number of time points to average over.
    n = data2.shape[1]

    rng = np.max(data2, axis=1) - np.min(data2, axis=1)
    try:
        d = np.sum(np.sqrt(1 / n * np.sum((data1 - data2) * (data1 - data2),
                                          axis=1)) / rng)
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    return d


def mean_absolute_error_dist(data1, data2):
    """Get the normalised manhattan distance between two numpy arrays.

    Parameters
    ----------
    data1 : np.ndarray
        First data array.

        The shape should match that of data2 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    data2 : np.ndarray
        Second data array.

        The shape should match that of data1 and the number of rows should
        match the number of model outputs i.e. 2 model outputs will be two
        rows.

    Returns
    -------
    d : float
        Normalised Manhattan distance measure

    """
    try:
        assert (data1.shape == data2.shape), 'Arrays not of equal size'
    except AssertionError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    assert (data1.shape == data2.shape), 'Arrays not of equal size'

    # Get number of time points to average over.
    n = data1.shape[1]

    try:
        d = 1 / n * np.sum(np.abs(data1 - data2))
    except ValueError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    return d


DISTANCES = {
    'euclidean': euclidean_dist,
    'manhattan': manhattan_dist,
    'MSE': mean_square_error_dist,
    'RMSE': root_mean_square_error_dist,
    'NRMSE': normalised_root_mean_square_error_dist,
    'MAE': mean_absolute_error_dist,
    'DTW': dtw_distance,
    'DTW.weighted': dtw_weighted_distance
}


def check_for_key(dictionary, target):
    """Check that a dictionary contains a key, and if so, return its data.

    Parameters
    ----------
    dictionary : dict
        Dictionary to check for `target` key.
    target : str
        String containing the target variable that is expected to be found in
        `dictionary`

    Returns
    -------
    data : list
        List of data found in `dictionary`. This is likely to be the time
        series data collected experimentally or generated by the model.

    """
    try:
        data = dictionary[target]
    except KeyError as e:
        print('Actual data does not contain target value of  {}'.format(target))
        raise e
    return data


def zero_array(array, zero_flag):
    """Zero an array of data with its initial values.

    Parameters
    ----------
    array : list
        List of data
    zero_flags : bool
        Boolean indicating if data needs zeroing
    Returns
    -------
    zerod : list
        Zero'd list

    """
    if zero_flag:
        init = float(array[0])
        zerod = [x - init for x in array]
    else:
        zerod = array
    return zerod


def get_distance(actual_data,
                 sim_data,
                 targets,
                 zero_flag,
                 distance='euclidean'):
    """Obtain  distance between two sets of data.

    Get a distance as defined by `distance` between two sets of data as well
    as between each signal in the data.

    Parameters
    ----------
    actual_data : dict
        Dictionary of actual data, as generated by
        :meth:`bayescmd.abc.data_import.import_actual_data`

    sim_data : dict
        Dictionary of simulated data, as created by
        :meth:`bayescmd.bcmdModel.ModelBCMD.output_parse`

    targets : list of :obj:`str`
        List of model targets, which should all be strings.

    zero_flag : dict
        Dictionary of form target(:obj:`str`): bool, where bool indicates
        whether to zero that target.

        Note: zero_flag keys should match targets list.

    distance : str, optional
        Name of distance measure to use. One of ['euclidean', 'manhattan',
        'MAE', 'MSE', 'NRMSE_range', 'NMRSE_mean'], where default is
        'euclidean'.

    Returns
    -------
    distances : dict
        Dictionary of form:
            {'TOTAL': summed distance of all signals,
            'target1: distance of 1st target',
            ...
            'targetN': distance of Nth target
            }

    """
    if (sorted(zero_flag.keys()) == sorted(targets)):
        d0 = []
        d_star = []
        for idx, k in enumerate(targets):
            try:
                d0.append(
                    zero_array(check_for_key(actual_data, k), zero_flag[k]))
            except (TypeError, IndexError):
                print('Invalid Data', end="\r")
                return (float('NaN'))
            try:
                d_star.append(
                    zero_array(check_for_key(sim_data, k), zero_flag[k]))
            except (TypeError, IndexError):
                print('Invalid Data', end="\r")
                return (float('NaN'))

        d0 = np.array(d0)
        d_star = np.array(d_star)

        distances = {"TOTAL": DISTANCES[distance](d_star, d0)}

        # Get distances for each individual signal
        for k in targets:
            d1 = zero_array(check_for_key(actual_data, k), zero_flag[k])
            d1 = np.array(d1).reshape(1, len(d1))

            d2 = zero_array(check_for_key(sim_data, k), zero_flag[k])
            d2 = np.array(d2).reshape(1, len(d2))

            distances[k] = DISTANCES[distance](d2, d1)
    else:
        zarray_err = """Targets doesn't match zero_flag dictionary.
        Targets: [{}]
        zflag: [{}]
        """.format(', '.join(['%s'] * len(targets)) % tuple(targets),
                   ', '.join(['%s'] * len(zero_flag.keys())) %
                   tuple(zero_flag.keys()))
        raise ZeroArrayError(zarray_err)

    return distances
