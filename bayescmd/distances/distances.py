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

        try:
            self.actual_data = np.array(check_for_key(
                import_actual_data(actual_data_file), target_value))
        except TypeError as e:
            print('Invalid File')
            print(e)

        self.sim_data = np.array(check_for_key(simulation_data, target_value))

    def euclidean_dist(self):
        return [np.sqrt(np.sum((self.actual_data - self.sim_data) *
                                     (self.actual_data - self.sim_data)))]
