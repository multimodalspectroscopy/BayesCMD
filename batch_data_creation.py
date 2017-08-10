"""
Class to create batch amounts of data from prior parameter distributions
"""

# import sys and os
import sys
sys.path.append('.')
import os
os.environ['BASEDIR'] = 'BayesCMD'
# import non custom packages
import numpy.random as rd
import numpy as np
from functools import partial
import math
import csv
import distutils.dir_util
import argparse
from subprocess import TimeoutExpired

# import bayescmd
from bayescmd.bcmdModel import ModelBCMD
import bayescmd.abc as abc
from bayescmd.util import findBaseDir


BASEDIR = findBaseDir(os.environ['BASEDIR'])


def returnConst(x):
    return x


class Batch:
    """
    This class will run a batch process of a rejection algorithm.
    """

    def __init__(self,
                 model_name,
                 prior_parameters,
                 inputs,
                 targets,
                 limit,
                 data_0,
                 workdir,
                 sample_rate=None):
        """
        Rejection will be used to run a simple ABC Rejection algorithm.

        Args:
            model_name (str): name of the model to simulate

            prior_parameters (dict): Dictionary of prior parameters. These take
            the form {"param":["prior_name", [*args]]}-args are prior specific

            epsilon (float): tolerance to keep/ignore parameter

            limit (int): Number of datasets to generate

            inputs (list): list of the dict keys with driving inputs

            targets (list): list of target simulation outputs

            data_0 (string): Path to csv of original experimental data

            workdir (str): file path to store all output data in.

            sample_rate (float): sample rate of data collected. Used only if time not provided in d0.
        """
        self.model_name = model_name

        self.priors = prior_parameters

        self.inputs = inputs

        self.targets = targets

        self.limit = limit

        self.d0 = abc.import_actual_data(data_0)

        self.workdir = workdir

        self.sample_rate = sample_rate

    @staticmethod
    def __getDistribution(v):
        """
        Get distribution from class prior selection.
        :param v: value from prior_parameters dict key:value pairing.
        :type v: list

        :return: Function that will generate a prior sample.
        :rtype: function
        """
        priorDict = {
            "constant": returnConst,
            "beta": rd.beta,
            "binomial": rd.binomial,
            "chisquare": rd.chisquare,
            "dirichlet": rd.dirichlet,
            "exponential": rd.exponential,
            "f": rd.f,
            "gamma": rd.gamma,
            "geometric": rd.geometric,
            "laplace": rd.laplace,
            "lognormal": rd.lognormal,
            "neg_binomial": rd.negative_binomial,
            "normal": rd.normal,
            "poisson": rd.poisson,
            "power": rd.power,
            "rayleigh": rd.rayleigh,
            "uniform": rd.uniform
        }
        return partial(priorDict[v[0]], *v[1])

    def definePriors(self):
        """
        Method to generate a dictionary of prior sampling functions for each
        parameter.
        """
        d = {}
        for k, v in self.priors.items():
            d[k] = self.__getDistribution(v)

        self.priorGen = d
        return d

    def generateOutput(self):
        params = {k: v() for k, v in self.priorGen.items()}
        data_length = max([len(l) for l in self.d0.values()])
        if ('t' not in self.d0.keys()) and (self.sample_rate is not None):
            times = [i * self.sample_rate for i in range(data_length + 1)]
        elif 't' in self.d0.keys():
            times = self.d0['t']
            if times[0] != 0:
                times.insert(0, 0)
        else:
            raise KeyError("time ('t') not in given data")
        if self.inputs is not None:
            inputs = abc.inputParse(self.d0, self.inputs)
        else:
            inputs = self.inputs

        abc_model = ModelBCMD(self.model_name,
                              inputs=inputs,
                              params=params,
                              times=times,
                              outputs=self.targets,
                              workdir=self.workdir,
                              suppress=True)
        if abc_model.debug:
            abc_model.write_initialised_input()
        try:
            abc_model.create_initialised_input()
            abc_model.run_from_buffer()
            output = abc_model.output_parse()
        except TimeoutExpired as e:
            output = None
        return params, output

    def batchCreation(self, zero_flag=None):
        STORE_VALUE = 1000
        prec_zero = max(2, int(math.log10(self.limit / STORE_VALUE)))
        parameters = []
        outputs = []
        distances = ['euclidean', 'manhattan', 'MSE', 'MAE']
        pf = os.path.join(self.workdir, "parameters.csv")
        header = [k for k in self.priors.keys()]
        header.insert(0, 'idx')
        header.extend(distances)
        for ii in range(self.limit):

            # ----- Write to file every STORE_VALUE entries ----- #
            if (ii % STORE_VALUE == 0) and (ii != 0):
                idx = str(int(ii / STORE_VALUE)).zfill(prec_zero)
                outf = os.path.join(self.workdir, "output_%s.csv" % (idx,))
                file_exists = os.path.isfile(pf)
                with open(pf, 'a') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=header)
                    if not file_exists:
                        writer.writeheader()
                    for idx, d in enumerate(parameters):
                        d['idx'] = idx
                        writer.writerow(d)

                with open(outf, 'w') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(outputs[0].keys())
                    for output in outputs:
                        writer.writerows(zip(*output.values()))
                outputs = []
                parameters = []

            elif (ii < STORE_VALUE) and (ii == self.limit - 1):
                outf = os.path.join(self.workdir, "output_00.csv")
                file_exists = os.path.isfile(pf)
                with open(pf, 'a') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=header)
                    if not file_exists:
                        writer.writeheader()
                    for idx, d in enumerate(parameters):
                        d['idx'] = idx
                        writer.writerow(d)

                with open(outf, 'w') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(outputs[0].keys())
                    for output in outputs:
                        writer.writerows(zip(*output.values()))
                outputs = []
                parameters = []
            elif (ii > STORE_VALUE) and (ii == self.limit - 1):
                idx = str(int(math.ceil(ii / STORE_VALUE))).zfill(prec_zero)
                outf = os.path.join(self.workdir, "output_%s.csv" % (idx,))
                file_exists = os.path.isfile(pf)
                out_exists = os.path.isfile(outf)
                with open(pf, 'a') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=header)
                    if not file_exists:
                        writer.writeheader()
                    for idx, d in enumerate(parameters):
                        d['idx'] = idx
                        writer.writerow(d)

                with open(outf, 'w') as out_file:
                    writer = csv.writer(out_file)
                    if not out_exists:
                        writer.writerow(outputs[0].keys())
                    for output in outputs:
                        writer.writerows(zip(*output.values()))
                outputs = []
                parameters = []

            params, output = self.generateOutput()

            if output is not None:
                output['ii'] = [ii] * len(output['t'])
                outputs.append(output)
                # ----- Add distances to the params dictionary ----- #
                for dist in distances:
                    params[dist] = abc.get_distance(self.d0,
                                                    output,
                                                    self.targets,
                                                    distance=dist,
                                                    zero_flag=zero_flag)
                parameters.append(params)
                print("Number of distances: {0:4d}".format(len(parameters)),
                      end="\r")

                sys.stdout.flush()

            else:
                print("TIMEOUT OCCURRED", end="\n")
                data_length = max([len(l) for l in self.d0.values()])
                output = {}
                output['ii'] = [ii] * data_length
                for t in self.targets:
                    output[t] = [np.nan] * data_length
                outputs.append(output)
                # ----- Add distances to the params dictionary ----- #
                for dist in distances:
                    params[dist] = np.nan
                parameters.append(params)
                print("Number of distances: {0:4d}".format(len(parameters)),
                      end="\r")

                sys.stdout.flush()

        return None

if __name__ == '__main__':

    ap = argparse.ArgumentParser('Choose model to batch run:')
    ap.add_argument('model', choices=['lv', 'bsx', 'BS'],
                    help='choice of model')
    ap.add_argument('run_length', metavar='RUN_LENGTH', type=int,
                    help='number of times to run the model')
    ap.add_argument('--test', metavar='RUN_LENGTH', action='store_true',
                    help='Whether to run this as a test')
    args = ap.parse_args()

    if args.test:
        if args.model == 'lv':
            model_name = 'lotka-volterra'
            inputs = None  # Input variables
            priors = {'a': ['uniform', [0, 2]],
                      'b': ['uniform', [0, 2]],
                      'c': ['uniform', [0, 2]],
                      'd': ['uniform', [0, 2]],
                      'x': ['constant', [80]],
                      'y': ['constant', [40]]}
            outputs = ['x', 'y']

            workdir = os.path.join(BASEDIR, 'build', 'batch', model_name)
            distutils.dir_util.mkpath(workdir)

            d0 = os.path.join(BASEDIR, 'build', 'lv_data.csv')

        elif args.model == 'bsx':
            model_name = 'bsx'
            inputs = ['Pa_CO2', 'P_a', 'u']  # Input variables
            priors = {'t_u': ['uniform', [0.4, 0.7]],
                      'v_un': ['uniform', [0.7, 1.3]]}
            outputs = ['HbO2', 'HHb']

            workdir = os.path.join(BASEDIR, 'build', 'batch', model_name)
            distutils.dir_util.mkpath(workdir)

            d0 = os.path.join(BASEDIR, 'data', 'bsxPFC.csv')

        elif args.model == 'BS':
            model_name = 'BS'
            inputs = ['Pa_CO2', 'P_a', 'u']  # Input variables
            priors = {'t_u': ['uniform', [0.4, 0.7]],
                      'v_un': ['uniform', [0.7, 1.3]]}
            outputs = ['HbO2', 'HHb']

            workdir = os.path.join(BASEDIR, 'build', 'batch', model_name)
            distutils.dir_util.mkpath(workdir)

            d0 = os.path.join(BASEDIR, 'data', 'bsxPFC.csv')

        else:
            print('Suitable test model not chosen')
            sys.exit(2)

        batchWriter = Batch(model_name,
                            priors,
                            inputs,
                            outputs,
                            args.run_length,
                            d0,
                            workdir)

        batchWriter.definePriors()
        batchWriter.batchCreation()
