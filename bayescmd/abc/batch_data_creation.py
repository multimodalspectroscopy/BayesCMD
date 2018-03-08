"""Create batch amounts of data from prior parameter distributions.

In order to undertake approximate bayesian computation, we need to be able to
run the model many times, drawing from the same prior distribution for
parameters. This module allows prior distributions to be specified and for the
model to be run multiple times, storing parameter combinations and the
resulting simulation's 'distance' from the true data, and also the simulated
data itself if wanted.

Attributes
----------
BASEDIR : :obj:`str`
    Path to Base directory, which should be 'BayesCMD'. It is found by running
    the :obj:`bayescmd.util.findBaseDir` method, passing either an environment
    variable or a string to the method.

"""

# import sys and os
import sys
# sys.path.append('.')
import os
# import non custom packages
import numpy.random as rd
import numpy as np
from functools import partial
import math
import csv
from subprocess import TimeoutExpired, CalledProcessError

from bayescmd.bcmdModel import ModelBCMD
from bayescmd.util import findBaseDir

from .data_import import inputParse
from .data_import import import_actual_data
from .distances import get_distance

os.environ['BASEDIR'] = 'BayesCMD'

BASEDIR = findBaseDir(os.environ['BASEDIR'])


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class ModelRunError(Error):
    """Class for if model run fails."""

    pass


def returnConst(x):
    """Return a constant value.

    This is for use with the prior generation method. The input value :obj:`x`
    will be returned, thus allowing a constant prior.

    Parameters
    ----------
    x : :obj:`float` or :obj:`int`
        Constant value to return.

    Returns
    -------
    x : :obj:`float`
        Return original input value.

    """
    return float(x)


class Batch:
    """Run a batch process of the rejection algorithm.

    The model will run a number of times, using configuration values specified
    to the class, drawing parameters from prior distributions.

    Parameters
    ----------
    model_name : :obj:`str`
        Name of the model to simulate. Should match the modeldef file for model
        being generated i.e. model_name of 'model`' should have a modeldef file
        'model1.modeldef'.

    prior_parameters : :obj:`dict`
        Dictionary of prior parameters. These take the form
        {"param"(:obj:`str`):["prior_name"(:obj:`str`), [*args]]} where the
        args are prior specific.

        Possible priors include:

        - constant
        - beta
        - binomial
        - chisquare
        - dirichlet
        - exponential
        - f
        - gamma
        - geometric
        - laplace
        - lognormal
        - neg_binomial
        - normal
        - poisson
        - power
        - rayleigh
        - uniform

        For information on the associated function and arguments, see
        :obj:`_getDistribution`.

    inputs : :obj:`list` of :obj:`str`
        List of the driving inputs. These should match column headings in the
        data file described by :obj:`data_0`.

    targets : :obj:`list` of :obj:`str`
        List of target simulation outputs.

    limit : :obj:`int`
        Number of times to run the model.

    data_0 : :obj:`str`
        Path to csv of original experimental data.

    workdir : :obj:`str`
        File path to the directory store all output data in. Output data will
        consist of a 'parameters.csv' file and, if :obj:`store_simulations` is
        :obj:`True`, then a numbered file containing each 1000 simulation
        outputs.

    store_simulations : :obj:`boolean`, optional
        Boolean flag for whether to store simulation outputs in addition to
        the parameters. Default is True.

    burnin : :obj:`int` or :obj:`float`, optional.
        Length of burn in period at start of the simulation. Default is 999.

    sample_rate : :obj:`float` or :obj:`None`, optional
        Sample rate of data collected. Used only if time not provided in
        data_0. Default is :obj:`None`.

    debug : :obj:`boolean`
        Indicates if debugging information should be written to console.
        Default is False.

    Attributes
    ----------
    model_name : :obj:`str`
        Name of the model to simulate. Should match the modeldef file for model
        being generated i.e. model_name of 'model`' should have a modeldef file
        'model1.modeldef'.

    priors : :obj:`dict`
        Dictionary of prior parameters. These take the form
        {"param"(:obj:`str`):["prior_name"(:obj:`str`), [*args]]} where the
        args are prior specific.

        Possible priors include:

        - constant
        - beta
        - binomial
        - chisquare
        - dirichlet
        - exponential
        - f
        - gamma
        - geometric
        - laplace
        - lognormal
        - neg_binomial
        - normal
        - poisson
        - power
        - rayleigh
        - uniform

        For information on the associated function and arguments, see
        :obj:`_getDistribution`.

    priorGen : :obj:`dict`
        Dictionary of functions that each sample from a prior distribution.

    inputs : :obj:`list` of :obj:`str`
        List of the driving inputs. These should match column headings in the
        data file described by :obj:`data_0`.

    targets : :obj:`list` of :obj:`str`
        List of target simulation outputs.

    limit : :obj:`int`
        Number of times to run the model.

    d0 : :obj:`dict`
        Dictionary of actual data. Keys are column headers, values are lists
        of data.

    workdir : :obj:`str`
        File path to the directory store all output data in. Output data will
        consist of a 'parameters.csv' file and, if :obj:`store_simulations` is
        :obj:`True`, then a numbered file containing each 1000 simulation
        outputs.

    store_simulations : :obj:`boolean`, optional
        Boolean flag for whether to store simulation outputs in addition to
        the parameters. Default is True.

    burnin : :obj:`int` or :obj:`float`, optional.
        Length of burn in period at start of the simulation. Default is 999.

    sample_rate : :obj:`float` or :obj:`None`, optional
        Sample rate of data collected. Used only if time not provided in
        data_0. Default is :obj:`None`.

    debug : :obj:`boolean`
        Indicates if debugging information should be written to console.
        Default is False.

    """

    def __init__(self,
                 model_name,
                 prior_parameters,
                 inputs,
                 targets,
                 limit,
                 data_0,
                 workdir,
                 store_simulations=True,
                 burnin=999,
                 sample_rate=None,
                 model_debug=False,
                 batch_debug=False,
                 autoParam=True,
                 offset=True):
        self.model_name = model_name

        self.priors = prior_parameters

        self.inputs = inputs

        self.targets = targets

        self.limit = limit

        self.d0 = import_actual_data(data_0)

        self.workdir = workdir

        self.store_simulations = store_simulations

        self.burnin = burnin

        self.sample_rate = sample_rate

        self.offset = offset

        self.model_debug = model_debug

        self.batch_debug = batch_debug

    @staticmethod
    def __getDistribution(v):
        """Get distribution from class prior.

        Parameters
        ----------
        v : :obj:`list`
            List of prior name and arguments. Prior name must be the first item
            in the list whilst the second should be an iterable containing
            the arguments for the distribution function.

        Returns
        -------
        :obj:`functools.partial`
        Function that will generate a prior sample.

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
        """Generate a :obj:`dict` of priors for each parameter.

        Creates a :obj:`dict` of functions that can be used to sample
        repeatedly from a prior distributions, with one for each parameter.

        Returns
        -------
        d : :obj:`dict`
            Dictionary of functions that each sample from a prior distribution.

        """
        d = {}
        for k, v in self.priors.items():
            d[k] = self.__getDistribution(v)

        self.priorGen = d
        return d

    def generateOutput(self):
        """Generate model output by drawing from a prior distribution.

        Returns
        -------
        (:obj:`dict` of 'parameters':values, :obj:`dict`of model outputs)
            Return a tuple of params, output.

        """

        if self.offset:
            for t in self.targets:
                self.priors["{}_offset".format(t)] = ["constant",
                                                      self.d0[t][0]]
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
            inputs = inputParse(self.d0, self.inputs)
        else:
            inputs = self.inputs

        abc_model = ModelBCMD(
            self.model_name,
            inputs=inputs,
            params=params,
            times=times,
            outputs=self.targets,
            workdir=self.workdir,
            debug=self.model_debug,
            burn_in=self.burnin,
            suppress=True)
        # if abc_model.debug:
        #     abc_model.write_initialised_input()
        try:
            abc_model.create_initialised_input()
            abc_model.run_from_buffer()
            output = abc_model.output_parse()
            if len(output['t']) == 0:
                raise ModelRunError

            for t in self.targets:
                try:
                    if np.isnan(np.array(output[t], dtype=np.float64)).any():
                        if self.batch_debug:
                            print('\nError in {}\n'.format(t))
                        raise ModelRunError
                except TypeError as e:
                    print(type(output[t]))
                    raise e
        except TimeoutExpired as e:
            if self.batch_debug:
                print("TIMEOUT OCCURRED", end="\n")
            output = None
        except ModelRunError as e:
            output = None
        except CalledProcessError as e:
            if self.batch_debug:
                print("Process failed to run correctly", end="\n")
            output = None
        return params, output

    def batchCreation(self, zero_flag):
        """Generate model outputs in a batch process.

        Model outputs will be written to file every 1000 runs if
        :obj:`store_simulations` is set to True, otherwise only parameters
        are written to file. Parameters are written to the same file,
        'parameters.csv' every 1000 runs.

        Four distances - euclidean, manhattan, mean absolute error (MAE) and
        mean squared error (MSE) - are written to parameters.csv also.

        All files are written into the directory defined by :obj:`workdir`.

        Parameters
        ----------
        zero_flag : :obj:`dict`
            Dictionary of form target(:obj:`str`): bool, where bool indicates
            whether to zero that target.

            Note: zero_flag keys should match targets list.

        Returns
        -------
        :obj:`None`
            Outputs are writen to file.

        """
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
                f_idx = str(int(ii / STORE_VALUE)).zfill(prec_zero)
                file_exists = os.path.isfile(pf)
                with open(pf, 'a') as param_file:
                    writer = csv.DictWriter(param_file, fieldnames=header)
                    if not file_exists:
                        writer.writeheader()
                    for idx, d in enumerate(parameters):
                        d['idx'] = idx + \
                            (int(ii / STORE_VALUE) - 1) * STORE_VALUE
                        writer.writerow(d)

                if self.store_simulations:
                    outf = os.path.join(self.workdir,
                                        "output_%s.csv" % (f_idx, ))
                    with open(outf, 'w') as out_file:
                        writer = csv.writer(out_file)
                        writer.writerow(outputs[0].keys())
                        for output in outputs:
                            writer.writerows(zip(*output.values()))
                outputs = []
                parameters = []

            elif (ii < STORE_VALUE) and (ii == self.limit - 1):

                file_exists = os.path.isfile(pf)
                with open(pf, 'a') as param_file:
                    writer = csv.DictWriter(param_file, fieldnames=header)
                    if not file_exists:
                        writer.writeheader()
                    for idx, d in enumerate(parameters):
                        d['idx'] = idx
                        writer.writerow(d)

                if self.store_simulations:
                    outf = os.path.join(self.workdir, "output_00.csv")
                    with open(outf, 'w') as out_file:
                        writer = csv.writer(out_file)
                        writer.writerow(outputs[0].keys())
                        for output in outputs:
                            writer.writerows(zip(*output.values()))
                outputs = []
                parameters = []

            elif (ii > STORE_VALUE) and (ii == self.limit - 1):
                f_idx = str(int(math.ceil(ii / STORE_VALUE))).zfill(prec_zero)

                file_exists = os.path.isfile(pf)
                with open(pf, 'a') as param_file:
                    writer = csv.DictWriter(param_file, fieldnames=header)
                    if not file_exists:
                        writer.writeheader()
                    for idx, d in enumerate(parameters):
                        d['idx'] = idx + STORE_VALUE * \
                            int(math.floor(ii / STORE_VALUE))
                        writer.writerow(d)

                if self.store_simulations:
                    outf = os.path.join(self.workdir,
                                        "output_%s.csv" % (f_idx, ))
                    out_exists = os.path.isfile(outf)
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
                    try:
                        cost = get_distance(
                            self.d0,
                            output,
                            self.targets,
                            distance=dist,
                            zero_flag=zero_flag)
                        params[dist] = cost['TOTAL']
                    except ValueError as error:
                        print("OUTPUT:\n", output)
                        raise error

                parameters.append(params)
                print(
                    "Number of distances: {0:4d}".format(len(parameters)),
                    end="\n")

                sys.stdout.flush()

            else:
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
                print(
                    "Number of distances: {0:4d}".format(len(parameters)),
                    end="\n")

                sys.stdout.flush()

        return None
