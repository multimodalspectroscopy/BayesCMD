# -*- coding: utf-8 -*-
"""Distances module.

This module is used to generate distance measures between simulated and real
time series.

Attributes
----------
DISTANCES : dict
    Dictionary contianing the distance aliases, mapping to the functions.

"""
import numpy as np
from numpy import AxisError
from scipy.stats import zscore

# All functions here can expect to handle the output from BCMD Model i.e.
# a dict.


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class ZeroArrayError(Error):
    """Exception raised for errors in the zero array."""

    pass


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
    except AxisError as e:
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
    except AxisError as e:
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
    except AxisError as e:
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
    except AxisError as e:
        print(e)
        print("\tData 1: ", data1.shape)
        print("\tData 2: ", data2.shape)
        raise e
    return d


DISTANCES = {
    'euclidean': euclidean_dist,
    'manhattan': manhattan_dist,
    'MSE': mean_square_error_dist,
    'MAE': mean_absolute_error_dist
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
    except KeyError:
        print('Actual data does not contain target value.')
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
                 distance='euclidean',
                 normalise=False):
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
        'MAE', 'MSE'], where default is 'euclidean'.

    normalise : bool, optional
        Boolean flag to indicate whether the signals need normalising, default
        is False. Current normalisation is done using z-score but that is
        likely to change with time.

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

        # Get distances for each individual signal
        for k in targets:
            d1 = zero_array(check_for_key(actual_data, k), zero_flag[k])
            d1 = np.array(d1).reshape(1, len(d1))

            d2 = zero_array(check_for_key(sim_data, k), zero_flag[k])
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
    else:
        zarray_err = """Targets doesn't match zero_flag dictionary.
        Targets: [{}]
        zflag: [{}]
        """.format(', '.join(['%s'] * len(targets)) % tuple(targets),
                   ', '.join(['%s'] * len(zero_flag.keys())) %
                   tuple(zero_flag.keys()))
        raise ZeroArrayError(zarray_err)

    return distances
