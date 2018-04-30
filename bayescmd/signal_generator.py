"""Generate simmulated input signals.

Generate simulated signals to use as inputs to the model.
"""
import numpy as np
import scipy.signal as sig
import numpy.random as rng


def rescale(vv, lo=0, hi=1):
    """
    Rescale a numpy array into the given range.

    Parameters
    ----------
    vv : :obj:`np.array`
        Numpy array to rescale.
    lo : :obj:`float`, optional
        Lowest value to scale to. Default is 0.
    hi : :obj:`float`, optional
        Highest value to scale to. Default is 1.

    Returns
    -------
    scaled_sig : :obj:`np.array`
        Rescaled Numpy array.

    """
    if lo is None or hi is None:
        scaled_sig = vv
    else:
        signal_min = np.min(vv)
        signal_max = np.max(vv)

        scaled_sig = lo + ((vv - signal_min) * (hi - lo) /
                           (signal_max - signal_min))
    return scaled_sig


def wave(tt, lo=0, hi=1, freq=0.1, sample_freq=1, phi=0, kind='saw', **kwargs):
    """
    Generate a single test waveform at specified time points.

    For a given set of times points, they are scaled to have a defined period
    of 2*pi

    Parameters
    ----------
    tt : :obj:`np.array`
        Numpy array of time points at which to generate signal.
    lo : :obj:`float`, optional
        Lowest value to scale to, if scaling. Default is 0.
    hi : :obj:`float`, optional
        Highest value to scale to, if scaling. Default is 1.
    freq : :obj:`float`, optional
        Frequency of the signal. Default is 0.1Hz.
    sample_freq : :obj:`float`, optional
        Sampling frequency of the time series. Default is 1.
    phi : :obj:`float`, optional
        Phase shift for rescaling time points. Default is 0.
    kind : :obj:`str`, optional
        Kind of wave to generate. Choice is:
            "saw" : sawtooth wave
            "square" : square wave
            "sine" : sinusoidal wave
            "uniform" : uniform wave
            "gaussian" : normally distributed random numbers
            "walk" : random walk
            "tophat" : single step function
        Each wave takes keyword arguments that can be passed through the
        function. For "tophat", the start and end are taken to be 25 and 75%
        of total end time respectively.
        Default wave is "saw".

    Returns
    -------
    :obj:`np.array`
        Recaled Numpy array of the chosen wave

    """
    # periodic signal functions have defined period 2 * pi, so we
    # scale the time points accordingly
    sct = (tt/sample_freq) * freq * 2 * np.pi + phi

    if kind == 'saw':
        if kwargs is not None and 'width' in kwargs:
            ss = sig.sawtooth(sct, width=kwargs['width'])
        else:
            ss = sig.sawtooth(sct)
    elif kind == 'square':
        if kwargs is not None and 'duty' in kwargs:
            ss = sig.square(sct, duty=kwargs['duty'])
        else:
            ss = sig.square(sct)
    elif kind == 'sine':
        ss = np.sin(sct)

    elif kind == 'uniform':
        ss = rng.rand(len(tt))
    elif kind == 'gaussian':
        ss = rng.randn(len(tt))
        if kwargs is not None:
            ss = ss * kwargs.get('sd', 1) + kwargs.get('mean', 0)
    elif kind == 'walk':
        ss = rng.randn(len(tt))
        if kwargs is not None:
            ss = ss * kwargs.get('sd', 1) + kwargs.get('mean', 0)
        ss = np.cumsum(ss)
    elif kind == 'tophat':
        if kwargs is not None and 'start' in kwargs:
            start = kwargs['start']
        else:
            start = 0.25

        if kwargs is not None and 'end' in kwargs:
            end = kwargs['end']
        else:
            end = 0.75

        ss = tt * 0
        ss[int(len(tt) * start):int(len(ss) * end)] = 1

    # potentially other kinds here

    # default to a constant 0 output
    else:
        ss = tt * 0

    return rescale(ss, lo, hi)


def waves(tt, specs=[{'kind': 'saw', 'width': 1}]):
    """
    Generate a signal that's the sum of multiple component waves at given timepoints.

    Parameters
    ----------
    tt : :obj:`np.array`
        Numpy array of time points at which to generate signal.
    specs: :obj:`dict` of form {'kind': :obj:`str`, **kwargs}, optional
        Dictionary specifying kind of wave, as well as any keyword arguments to
        the :obj:`wave` function

    Returns
    -------
    result : :obj:`np.array`'width':
        Numpy array containing the sum of each wave type specified in specs
        at the time points specified by :param:tt

    """
    result = 0
    for spec in specs:
        if spec['kind'] == 'rescale':
            result = rescale(result, **spec)
        else:
            result = result + wave(tt, **spec)
    return result


def generate(n=20, timescale=1, start=0, lo=0, hi=1,
             specs=[{'kind': 'saw'}]):
    """
    Generate a wave between given time points, using given parameters.

    Parameters
    ----------
    n : :obj:`int`, optional
        Number of time points. Default is 20.
    timescale: obj:`float`, optional
        Timescale for points e.g. 1 second, 0.1 second etc. Default is 1.
    start : :obj:`float`
        Starting time. Default is 0.
    lo : :obj:`float`, optional
        Lowest value to scale overall wave to. Default is 0.
    hi : :obj:`float`, optional
        Highest value to scale overall wave to. Default is 1.
    specs : :obj:`list` of spec :obj:`dict`
        List of wave specification dictionaries. These can contain individual
        wave lower and upper bounds to scale to, wave phase shifts, wave
        frequencies

        Default is {'kind': 'saw'}

    Returns
    -------
    :obj:`dict` of form {'t': :obj:`np.array`, 'signal', :obj:`np.array`},
    where 't' is the time data and 'signal' is the generated signal.

    """
    tt = np.linspace(start=start, stop=start + (n - 1) * timescale, num=n)
    result = waves(tt, specs)
    return {'t': tt, 'signal': rescale(result, lo, hi)}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # simple test driver -- redirect to file to examine
    sp1 = {'kind': 'sine', 'lo': -10, 'hi': 10, 'freq': 10}
    sp2 = {'kind': 'walk', 'sd': 0.1, 'lo': None}
    sp3 = {'kind': 'square', 'lo': 0, 'hi': 10, 'freq': 3}
    sp4 = {'kind': 'gaussian', 'sd': 0.5, 'lo': None}
    wv = generate(n=1000, timescale=0.001, specs=[
                  sp1, sp2, sp3, sp4], lo=0, hi=5)
    plt.plot(wv['t'], wv['signal'])
    plt.show()
