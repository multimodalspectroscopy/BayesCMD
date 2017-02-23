import numpy.random as rd
from functools import partialmethod


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

    def __init__(self, prior_parameters):
        """
        Rejection will be used to run a aimple ABC Rejection algorithm.

        Args:
            prior_parameters (dict): Dictionary of prior parameters. These take
            theform {"param":["prior_name", [*args]]}-args are prior specific.
        """
    self.priors = prior_parameters

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

    def generateModel(self):
        params = {k: v() for k, v in self.priorGen.items()}
        abc_model = ModelBCMD(model,
                              inputs,
                              params,
                              times)
        abc_model.create_default_input()
        abc_model.run_from_buffer()
        output = abc_model.output_parse()
