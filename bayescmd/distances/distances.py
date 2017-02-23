import numpy as np
from .data_import import *
# All functions here can expect to handle the output from BCMD Model i.e.
# a dict.


def check_for_key(dictionary, target):
    try:
        data = dictionary[target]
    except KeyError:
        print('Actual data does not contain target value.')
    return data


class DistanceMeasures:

    def __init__(self,
                 target_value,
                 simulation_data,
                 actual_data_file):

        self.actual_data = check_for_key(
            actual_data_import(actual_data_file), target_value)

        self.sim_data = check_for_key(sim_data, target_value)

    def euclidean_dist():
        return [numpy.sqrt(numpy.sum((self.actual_data - self.sim_data) *
                                     (self.actual_data - self.sim_data)))]
