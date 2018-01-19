"""Run a simple hypercapnia batch using the BS model."""
from bayescmd.abc import Batch
from datetime import datetime
import os
# import sys
import pprint
import distutils.dir_util
import argparse
import json
from bayescmd.util import findBaseDir
from bayescmd.abc import priors_creator
BASEDIR = findBaseDir('BayesCMD')

# model_name = 'BS'
# inputs = ['Pa_CO2', 'P_a', 'SaO2sup']  # Input variables
#
# priors = priors_creator({
#     "Vol_mit": 0.067,
#     "r_t": 0.018,
#     "r_m": 0.027,
#     "r_0": 0.0126,
#     "O2_n": 0.024,
#     "cytox_tot_tis": 0.0055,
#     "v_cn": 40,
#     "sigma_coll": 62.79
# }, 0.25)
# outputs = ['Vmca', 'CCO']


def process(conf, run_length, data_0, workdir, debug=False):
    """
    Rejection will be used to run a simple ABC Rejection algorithm.

    Args:
        conf (dict): Dict containing the following parameters:

            model_name (str): name of the model to simulate

            parameters (dict): Dictionary of parameters. These take
            the form {"param": param_value}

            inputs (list): list of the driving inputs

            targets (list): list of target simulation outputs

            burnin (int): Number of seconds to burn in for

            sample_rate (float): sample rate of data collected. Used only
                if time not provided in d0.

            debug (bool): Whether to run this in debug mode.

        data_0 (string): Path to csv of original experimental data

        workdir (str): file path to store all output data in.
    """
    model_name = conf['model_name']

    priors = priors_creator(conf['priors']['defaults'],
                            conf['priors']['variation'])

    inputs = conf['inputs']

    targets = conf['targets']

    d0 = data_0

    workdir = workdir

    if 'debug' in conf.keys() and debug is False:
        debug = conf['debug']
    else:
        debug = debug

    if debug:
        print("##### PRIORS ####")
        pprint.pprint(priors)

    batchWriter = Batch(
        model_name,
        priors,
        inputs,
        targets,
        run_length,
        d0,
        workdir,
        store_simulations=True,
        debug=debug)

    batchWriter.definePriors()
    batchWriter.batchCreation(zero_flag={k: False for k in targets})


if __name__ == '__main__':
    ap = argparse.ArgumentParser('Choose batch run data and length:')
    ap.add_argument(
        'run_length',
        metavar='RUN_LENGTH',
        type=int,
        help='number of times to run the model')
    ap.add_argument('input_file', metavar="INPUT_FILE", help='batch data')
    ap.add_argument('conf', help='Configuration file: Should be JSON')
    ap.add_argument(
        '--workdir',
        help='Directory to store output (optional).',
        default=None)
    ap.add_argument(
        '--debug',
        help='Flag to show debug info (optional).',
        action='store_true')

    args = ap.parse_args()
    now = datetime.now().strftime('%H%MT%d%m%y')
    with open(args.conf, 'r') as conf_f:
        config = json.load(conf_f)
    if args.workdir is None:
        workdir = os.path.join(BASEDIR, 'build', 'batch',
                               config['model_name'], now)
    else:
        workdir = args.workdir
    distutils.dir_util.mkpath(workdir)
    process(config, args.run_length, args.input_file, workdir, args.debug)
