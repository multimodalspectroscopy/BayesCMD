import numpy.random as rd
from functools import partialmethod
from .distances import get_distance
from .data_import import *


class Rejection:
    """
    This class will run a batch process of a rejection algorithm.
    """
    priorDict = {
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

    def __init__(self,
                 model_name,
                 prior_parameters,
                 epsilon,
                 n_particles,
                 inputs,
                 targets,
                 data_0,
                 particle_limit=10000,
                 sample_rate=None):
        """
        Rejection will be used to run a simple ABC Rejection algorithm.

        Args:
            model_name (str): name of the model to simulate

            prior_parameters (dict): Dictionary of prior parameters. These take
            the form {"param":["prior_name", [*args]]}-args are prior specific

            epsilon (float): tolerance to keep/ignore parameter

            n_particles (int): target num of particles to construct posterior

            inputs (list): list of the dict keys with driving inputs

            targets (list): list of target simulation outputs

            data_0 (string): Path to csv of original experimental data

            particle_limit (int): limit at which to stop constructing posterior
        """
    self.priors = prior_parameters

    self.eps = epsilon

    self.n_particles = n_particles

    self.limit = particle_limit

    self.d0 = import_actual_data(data_0)

    @staticmethod
    def __getDist(v):
        """
        Get distribution from class prior selection.
        :param v: value from prior_parameters dict key:value pairing.
        :type v: list

        :return: Function that will generate a prior sample.
        :rtype: function
        """
        yield partialmethod(priorDict[v[0]], *v[1])

    def definePriors(self):
        """
        Method to generate a dictionary of prior sampling functions for each
        parameter.
        """
        d = {}
        for k, v in self.priors.items():
            d[k] = self.__getDist(v)

        self.priorGen = d
        return d

    def generateOutput(self):
        params = {k: v() for k, v in self.priorGen.items()}
        data_length = max([len(l) for l in self.d0.values()])
        if 't' not in self.d0.keys() and sample_rate not None:
            times = [i * sample_rate for i in range(data_length)]
        elif 't' in self.d0.keys():
            times = self.d0['t']
        else:
            raise KeyError("time ('t') not in given data")

        inputs = self.d0[self.inputs]

        abc_model = ModelBCMD(self.model_name,
                              inputs,
                              params,
                              times)
        abc_model.create_default_input()
        abc_model.run_from_buffer()
        output = abc_model.output_parse()
        return params, output

    def rejectionAlg(self):

        self.posterior = {k: [] for k in self.priors.keys()}

        for ii in range(self.limit):
            currParticles = len(list(self.posterior.values())[0])
            if currParticles <= self.n_particles:
                params, output = self.generateOutput()
                if get_distance(self.d0, output, self.targets) <= self.eps:
                    for k, v in params.items():
                        self.posterior[k].extend(v)
                else:
                    pass
            else:
                break

        return self.posterior
