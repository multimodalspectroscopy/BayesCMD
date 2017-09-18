from math import log10, floor


def round_sig(x, sig=1):
    """
    Round a value to N sig fig
    Inputs:
    =======
    x: float to round
    sig: integer number of sig figs

    Outputs:
    ========
    Rounded float
    """
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def priors_creator(params, variation, dist_type='uniform'):
    """
    Function create an input dictionary of uniform parameter priors.
    Inputs:
    =======
    params: dictionary of paramaters and their default value
    variation: float above 0 to define limits of distribution.
    """
    return {k: [dist_type, [v * (1 - variation), v * (1 + variation)]]
            for k, v in params.items()}


def openopt_param_creator(d, f):
    """
    Function create an input dictionary of uniform parameter priors.
    Inputs:
    =======
    d: dictionary of paramaters and their default value
    f: float above 0 to define limits of distribution.
    """
    for k, v in d.items():
        print('param: {}, uniform, {:.4f}, {:.4f}, {:.4f}'.format(
            k, (1 - f) * v, (1 + f) * v, v))
        return None
