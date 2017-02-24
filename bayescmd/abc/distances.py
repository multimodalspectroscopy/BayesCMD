import numpy as np
from .data_import import *
# All functions here can expect to handle the output from BCMD Model i.e.
# a dict.


def euclidean_dist(data1, data2):
    return [np.sqrt(np.sum((data1 - data2) * (data1 - data2)))]


DISTANCES = {
    'euclidean': euclidean_dist
}


def check_for_key(dictionary, target):
    try:
        data = dictionary[target]
    except KeyError:
        print('Actual data does not contain target value.')
    return data


def get_distance(actual_data, sim_data, targets, distance='euclidean'):

    d0 = []
    d_star = []
    for idx, k in enumerate(targets):
        print('INDEX: ', idx)
        d0.append(check_for_key(actual_data, k))
        d_star.append(check_for_key(sim_data, k))

    return DISTANCES[distance](np.array(d0), np.array(d_star))
