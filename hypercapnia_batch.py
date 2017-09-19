from batch_data_creation import Batch
from datetime import datetime
import os
import sys
import pprint
import distutils
import argparse
from bayescmd.util import findBaseDir
from bayescmd.abc import priors_creator
BASEDIR = findBaseDir('BayesCMD')

model_name = 'BS'
inputs = ['Pa_CO2', 'P_a', 'SaO2sup']  # Input variables

priors = priors_creator({"Vol_mit": 0.067,
                         "r_t": 0.018,
                         "r_m": 0.027,
                         "r_0": 0.0126,
                         "O2_n": 0.024,
                         "cytox_tot_tis": 0.0055,
                         "v_cn": 40,
                         "sigma_coll": 62.79}, 0.25)

# priors = {"Vol_mit": ['uniform', [0.02, 0.12]],
#           'r_t': ['uniform', [0.01, 0.03]],
#           'r_m': ['uniform', [0.01, 0.04]],
#           'r_0': ['uniform', [0.007, 0.0175]],
#           'cytox_tot_tis': ['uniform', [0.0025, 0.009]]
#           'v_cn': ['uniform', [28, 52]]
#           'sigma_coll':
#           'O2_n':
#           }
# test_priors = {"Vol_mit": ['uniform', [0.088, 0.09]],
#                'r_t': ['uniform', [0.0165, 0.017]],
#                'r_m': ['uniform', [0.0315, 0.035]],
#                'r_0': ['uniform', [0.012, 0.0124]],
#                'cytox_tot_tis': ['uniform', [0.00616, 0.0064]]
#                }
outputs = ['Vmca', 'CCO']


def process(run_length, input_file, workdir):
    batchWriter = Batch(model_name,
                        priors,
                        inputs,
                        outputs,
                        run_length,
                        input_file,
                        workdir)

    batchWriter.definePriors()
    batchWriter.batchCreation(zero_flag=None)

if __name__ == '__main__':
    ap = argparse.ArgumentParser('Choose batch run data and length:')
    ap.add_argument('input_file', metavar="INPUT_FILE",
                    help='batch data')
    ap.add_argument('run_length', metavar='RUN_LENGTH', type=int,
                    help='number of times to run the model')

    args = ap.parse_args()
    now = datetime.now().strftime('%H%MT%d%m%y')

    print("##### PRIORS ####")
    pprint.pprint(priors)
    workdir = os.path.join(BASEDIR, 'build', 'batch', model_name, now)
    distutils.dir_util.mkpath(workdir)
    process(args.run_length, args.input_file, workdir)
