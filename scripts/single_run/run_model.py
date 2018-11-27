from bayescmd.util import findBaseDir
import bayescmd.abc as abc
from bayescmd.bcmdModel import ModelBCMD
import itertools
import numpy as np
from datetime import datetime
import distutils.dir_util
from subprocess import TimeoutExpired
import argparse
import csv
import json
import os
"""Class to run model once."""

# import sys and os
import sys
sys.path.append('..')
#os.environ['BASEDIR'] = 'BayesCMD'
# import non custom packages
# import bayescmd

BASEDIR = findBaseDir(os.environ['BASEDIR'])
print(BASEDIR)


class RunModel:
    def __init__(self,
                 conf,
                 data_0,
                 workdir,
                 burnin=999,
                 sample_rate=None,
                 debug=False):
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
        self.model_name = conf['model_name']

        self.params = conf['parameters']

        self.inputs = conf['inputs']

        self.targets = conf['targets']

        self.d0 = abc.import_actual_data(data_0)

        self.workdir = workdir

        if 'burnin' in conf.keys():
            self.burnin = conf['burnin']
        else:
            self.burnin = burnin
        if 'sample_rate' in conf.keys():
            self.sample_rate = conf['sample_rate']
        else:
            self.sample_rate = sample_rate
        if 'debug' in conf.keys() and debug is False:
            self.debug = conf['debug']
        else:
            self.debug = debug
        if 'outf' in conf.keys():
            self.outf = conf['outf']
        else:
            self.outf = None

    def generateOutput(self):
        data_length = max([len(l) for l in self.d0.values()])
        if ('t' not in self.d0.keys()) and (self.sample_rate is not None):
            times = [i * self.sample_rate for i in range(data_length + 1)]
        elif 't' in self.d0.keys():
            times = self.d0['t']
            if times[0] != 0:
                times.insert(0, 0)
        else:
            raise KeyError(
                "time ('t') not in given data and no sample rate provided")
        if self.inputs is not None:
            inputs = abc.inputParse(self.d0, self.inputs)
        else:
            inputs = self.inputs

        model = ModelBCMD(
            self.model_name,
            inputs=inputs,
            params=self.params,
            times=times,
            outputs=self.targets,
            workdir=self.workdir,
            debug=self.debug,
            burn_in=self.burnin,
            suppress=True)

        if model.debug:
            model.write_initialised_input()
        try:
            model.create_initialised_input()
            model.run_from_buffer()
            output = model.output_parse()
        except TimeoutExpired as e:
            output = None

        self.output = output
        return output

    def writeOutput(self):
        if self.outf:
            outf = os.path.join(self.workdir,
                                "model_run_output_{}.csv".format(self.outf))
        else:
            now = datetime.now().strftime("%d%m%yT%H%M")
            outf = os.path.join(self.workdir,
                                "model_run_output_{}.csv".format(now))

        with open(outf, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(self.output.keys())
            writer.writerows(zip(*self.output.values()))

        print("output written to {}".format(outf))


if __name__ == '__main__':

    ap = argparse.ArgumentParser('Choose configuration file:')
    ap.add_argument(
        'input_file',
        metavar="INPUT_FILE",
        help='CSV input file from experiment/simulation')
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
    now = datetime.now().strftime('%d%m%yT%H%M')

    with open(args.conf, 'r') as conf_f:

        config = json.load(conf_f)

    if args.workdir is None:
        workdir = os.path.join(BASEDIR, 'build', 'batch', config['model_name'],
                               now)
    else:
        workdir = os.path.join(args.workdir, config['model_name'], now)

    distutils.dir_util.mkpath(workdir)

    model = RunModel(
        conf=config, data_0=args.input_file, workdir=workdir, debug=args.debug)
    output = model.generateOutput()
    model.writeOutput()
