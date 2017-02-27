from bayescmd.abc.data_import import *
from bayescmd.abc.distances import get_distance
from bayescmd.abc.rejection import Rejection
from bayescmd.bcmdModel import ModelBCMD

from nose.tools import assert_true, assert_equal, with_setup, assert_dict_equal
import numpy.testing as np_test

import os

BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
assert os.path.basename(BASEDIR) == 'BayesCMD'
print(BASEDIR)


def test_csv_data_import():
    """
    Check that an experimental data is processed correctly.
    """
    expt_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'test_files', 'rc_actual_data.csv')

    expt_data = import_actual_data(expt_file)

    test_dict = {'V': [1.0, 0.0, -1.0, 0.0, 1.0],
                 'Vc': [0.99326202,
                        0.00669255,
                        -0.99321691,
                        -0.00669224,
                        0.99321693],
                 't': [5.0, 10.0, 15.0, 20.0, 25.0]}

    assert_dict_equal(expt_data, test_dict)


class TestDistances_Single:

    def setUp(self):
        self.x1 = {'data': [1, 2, 3]}
        self.x2 = {'data': [4, 5, 6]}

    def test_euclidean_distance(self):
        distance = get_distance(self.x1, self.x2, ['data'], 'euclidean')

        np_test.assert_almost_equal([5.196152422], distance)

    def test_manhattan_distance(self):

        distance = get_distance(self.x1, self.x2, ['data'], 'manhattan')

        np_test.assert_almost_equal([9.0], distance)

    def test_MSE_distance(self):

        distance = get_distance(self.x1, self.x2, ['data'], 'MSE')

        np_test.assert_almost_equal([9.0], distance)

    def test_MAE_distance(self):

        distance = get_distance(self.x1, self.x2, ['data'], 'MAE')

        np_test.assert_almost_equal([3.0], distance)


class TestDistances_Multiple:

    def setUp(self):
        self.x1 = {'data1': [1, 2, 3],
                   'data2': [4, 5, 6]}
        self.x2 = {'data1': [2, 3, 4],
                   'data2': [7, 8, 9]}

    def test_euclidean_distance(self):
        distance = get_distance(self.x1, self.x2, ['data1', 'data2'],
                                'euclidean')

        np_test.assert_almost_equal([6.92820323027], distance)

    def test_manhattan_distance(self):

        distance = get_distance(self.x1, self.x2, ['data1', 'data2'],
                                'manhattan')

        np_test.assert_almost_equal([12.0], distance)

    def test_MSE_distance(self):

        distance = get_distance(self.x1, self.x2, ['data1', 'data2'],
                                'MSE')

        np_test.assert_almost_equal([10.0], distance)

    def test_MAE_distance(self):

        distance = get_distance(self.x1, self.x2, ['data1', 'data2'],
                                'MAE')

        np_test.assert_almost_equal([4.0], distance)


class TestModelDistances():

    def setUp(self):
        # We define R here to be not the default in order to generate test
        # data appropriately.
        self.name = 'rc'
        self.priors = {"C": ['uniform', [1e-3, 1]],
                       "R": ['constant', [1000]]}
        self.epsilon = 1e-1
        self.n_particles = 1000
        self.inputs = ['V']
        self.targets = ['Vc']
        self.limit = int(1e8)
        self.data = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 'test_files', 'rc_actual_data.csv')

    def test_rejection(self):

        testReject = Rejection(model_name=self.name,
                               prior_parameters=self.priors,
                               epsilon=self.epsilon,
                               n_particles=self.n_particles,
                               inputs=self.inputs,
                               targets=self.targets,
                               particle_limit=self.limit,
                               data_0=self.data)

        testReject.definePriors()
        post = testReject.rejectionAlg()
        print(post)
