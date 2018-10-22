"""Run a simple hypercapnia batch using the BS model."""
# import sys and os
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
print(sys.path)
sys.path.append(os.environ['HOME'] + '/BayesCMD')
from bayescmd.abc import Batch
from datetime import datetime
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


def process(conf, run_length, data_0, workdir, batch_debug=False,
            model_debug=False, offset=True):
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

            batch_debug (bool): Whether to run this in batch debug mode.

            model_debug (bool): Whether to run this in model debug mode.

            store_simulations (bool): Whether to store simulations as well as
                parameters

        data_0 (string): Path to csv of original experimental data

        workdir (str): file path to store all output data in.

        offset (bool): Whether to set target offsets to first value.
    """
    model_name = conf['model_name']

    if conf['create_params']:
        priors = priors_creator(conf['priors']['defaults'],
                                conf['priors']['variation'])
    else:
        priors = conf['priors']

    inputs = conf['inputs']

    targets = conf['targets']

    d0 = data_0

    if 'store_simulations' not in conf.keys():
        conf['store_simulations'] = True

    workdir = workdir

    if 'batch_debug' in conf.keys() and batch_debug is False:
        batch_debug = conf['batch_debug']
    else:
        batch_debug = batch_debug

    if 'model_debug' in conf.keys() and model_debug is False:
        model_debug = conf['model_debug']
    else:
        model_debug = model_debug

    if model_debug:
        print("##### PRIORS ####")
        pprint.pprint(priors)

    if "zero_flag" in conf.keys():
        zero_flag = conf['zero_flag']
    else:
        zero_flag = {k: False for k in targets}

    batchWriter = Batch(
        model_name,
        priors,
        inputs,
        targets,
        run_length,
        d0,
        workdir,
        store_simulations=conf['store_simulations'],
        offset=True,
        batch_debug=batch_debug,
        model_debug=model_debug)

    batchWriter.definePriors()
    batchWriter.batchCreation(zero_flag= zero_flag)


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
