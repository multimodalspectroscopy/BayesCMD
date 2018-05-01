from batch_data_creation import Batch
from datetime import datetime
import os
import sys
import distutils
import argparse
from bayescmd.util import findBaseDir
BASEDIR = findBaseDir('BayesCMD')

model_name = 'BS'
inputs = ['u']  # Input variables
priors = {'r_t': ['uniform', [0.009, 0.027]],
          'r_m': ['uniform', [0.0135, 0.0405]],
          'r_0': ['uniform', [0.0063, 0.019]]
          }
outputs = ['CCO']



def process(run_length, input_file, workdir):
    batchWriter = Batch(model_name,
                        priors,
                        inputs,
                        outputs,
                        run_length,
                        input_file,
                        workdir)

    batchWriter.definePriors()
    batchWriter.batchCreation(zero_flag=[0])

if __name__ == '__main__':
    ap = argparse.ArgumentParser('Choose model to batch run:')
    ap.add_argument('input_file', metavar="INPUT_FILE", help='choice of model')
    ap.add_argument('run_length', metavar='RUN_LENGTH', type=int,
                    help='number of times to run the model')

    args = ap.parse_args()
    now = datetime.now().strftime('%H%MT%d%m%y')

    workdir = os.path.join(BASEDIR, 'build', 'batch', model_name, now)
    distutils.dir_util.mkpath(workdir)
    process(args.run_length, args.input_file, workdir)