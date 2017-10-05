"""Create priors for use in BayesCMD and openopt."""


def priors_creator(params, variation, dist_type='uniform'):
    """Create an input dictionary of uniform parameter priors.

    Parameters
    ----------
    params : dict
        Dictionary of form {'parameter': default value}

    variation : float
        Float greater than 0 to define limits of distribution. A value of 0.25
        is equivalent to a 25% change either way.

    dist_type : str, optional
        Distribution type for parameter. At present, only 'uniform' is valid.

    Returns
    -------
    dict
        Dict of parameters with prior specification. Should have form:
        {'parameter': [distribution_type, [lower_bound, upper_bound]]}

    """
    if dist_type is not 'uniform':
        raise ValueError('InValid Distribution Type: {}'.format(dist_type))

    return {
        k: [dist_type, [v * (1 - variation), v * (1 + variation)]]
        for k, v in params.items()
    }


def openopt_param_creator(d, f):
    """Create an input dictionary of uniform parameter priors for openopt.

    Parameters
    ----------
    d : dict
        Dictionary of paramaters and their default values.

    f : float
        Float greater than 0 to define limits of distribution. A value of 0.25
        is equivalent to a 25% change either way.

    Returns
    -------
    None
        Prints parameter specification to terminal in format compatible with
        :module:`optim.py`

    """
    for k, v in d.items():
        print('param: {}, uniform, {:.4f}, {:.4f}, {:.4f}'.format(
            k, (1 - f) * v, (1 + f) * v, v))
        return None
