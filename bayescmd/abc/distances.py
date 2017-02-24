import numpy as np
from .data_import import *
# All functions here can expect to handle the output from BCMD Model i.e.
# a dict.

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

    actual_data = np.array([])
    sim_data = np.array([])
    for idx, k in enumerate(targets):
        d0[idx, :] = np.array(check_for_key(actual_data, target_value))
        d_star[idx, :] = np.array(check_for_key(sim_data, target_value))

    return DISTANCES[distance](d0, d_star)


def euclidean_dist(data1, data2):
    return [np.sqrt(np.sum((data1 - data2) * (data1 - data2)))]
