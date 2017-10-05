"""Miscellaneous utility functions used throughout BayesCMD.

This module contains a number of utility functions that are used throughout
the different BayesCMD subpackages.
"""
import os
import sys
from math import log10, floor


def round_sig(x, sig=1):
    """Round a value to N sig fig.

    Parameters
    ----------
    x : float
        Value to round
    sig : int, optional
        Number of sig figs, default is 1

    Returns
    -------
    float
        Rounded value

    """
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def findBaseDir(basename, max_depth=5, verbose=False):
    """
    Get relative path to a BASEDIR.
    :param basename: Name of the basedir to path to
    :type basename: str

    :return: Relative path to base directory.
    :rtype: StringIO
    """
    MAX_DEPTH = max_depth
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    print("STARTING AT: %s\n Looking for: %s" % (BASEDIR, basename))
    for level in range(MAX_DEPTH):
        if verbose:
            print('LEVEL %d: %s\n Current basename: %s' %
                  (level, BASEDIR, os.path.basename(BASEDIR)))
        if os.path.basename(BASEDIR) == basename:
            break
        else:
            BASEDIR = os.path.abspath(os.path.dirname(BASEDIR))
        if level == MAX_DEPTH - 1:
            sys.exit(
                'Could not find correct basedir\n Currently at %s' % BASEDIR)
    return os.path.relpath(BASEDIR)
