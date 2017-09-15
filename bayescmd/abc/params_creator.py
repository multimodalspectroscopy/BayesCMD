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
