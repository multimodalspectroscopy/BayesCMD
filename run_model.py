"""Class to run model once."""

# import sys and os
import sys
sys.path.append('.')
import os
#os.environ['BASEDIR'] = 'BayesCMD'
# import non custom packages
import json
import csv
import argparse
from subprocess import TimeoutExpired
import distutils.dir_util
from datetime import datetime
import numpy as np
import itertools
# import bayescmd
from bayescmd.bcmdModel import ModelBCMD
import bayescmd.abc as abc
from bayescmd.util import findBaseDir

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
        outf = os.path.join(self.workdir, "model_run_output.csv".format(now))

        with open(outf, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(self.output.keys())
            writer.writerows(zip(*self.output.values()))

        print("output written to {}".format(outf))


class RunSteadyState:
    def __init__(self,
                 conf,
                 workdir,
                 debug=False):
        """
        Run a simple steady state simulation of a model.

        Args:
            conf (dict): Dict containing the following parameters:

                model_name (str): name of the model to simulate

                parameters (dict): Dictionary of parameters. These take
                the form {"param": param_value}

                inputs (list): list of the driving inputs

                targets (list): list of target simulation outputs

                debug (bool): Whether to run this in debug mode.

            data_0 (string): Path to csv of original experimental data

            workdir (str): file path to store all output data in.
        """
        self.modelName = conf['model_name']

        self.params = conf['parameters']

        self.inputs = conf['inputs']

        self.outputs = conf['targets']

        self.outputs.append(self.inputs)

        self.max = conf['max_val']

        self.min = conf['min_val']

        if 'direction' in conf.keys():
            self.direction = conf['direction']
        else:
            self.direction = "up"

        self.workdir = workdir

        if 'debug' in conf.keys() and debug is False:
            self.debug = conf['debug']
        else:
            self.debug = debug

    @staticmethod
    def make_steady_model(modelName,
                          inputs,
                          outputs,
                          max_val,
                          min_val,
                          params={},
                          direction='up'):
        """
        Create a model that will run a steady state simulation.

        Parameters
        ----------
            modelName : str
                Name of model to run.

            inputs : str
                Input variable that is being varied.

            outputs : list
                List of target simulation outputs.

            max_val : float
                Maximum value of input.

            min_val : float
                Minimum value of input.

            params : dict, optional
                Dictionary of parameters. These take the form
                {"param": param_value}

            direction : str
                Direction to run steady state simulation in. Should be one of
                "up", "down" or "both".

        Returns
        -------


        """
        def _reorder(_min, _max):
            """
            Ensures correct ordering of min_val and max_val

            Parameters
            ----------
            _min : float
                Intended minimum value
            _max : float
                Intended maximum value

            Returns
            -------
            tuple
                Tuple ordered such that first item will always be the minimum
            """
            if _max < _min:
                return (_max, _min)
            else:
                return (_min, _max)

        min_val, max_val = _reorder(min_val, max_val)

        dirs = ['up', 'down', 'both']
        if direction not in dirs:
            raise ValueError("Invalid direction. Expected one of: %s" % dirs)

        steady_input = {'names': [inputs]}

        steps = np.array(
            list(
                itertools.chain(
                    *[[x] * 100 for x in np.linspace(min_val, max_val, 50)])))

        if direction == 'up':
            steady_input['values'] = steps.reshape(len(steps), 1)

        elif direction == 'down':
            steady_input['values'] = steps[::-1].reshape(len(steps), 1)

        elif direction == 'both':
            x = np.concatenate((steps, steps[-2::-1]))
            steady_input['values'] = x.reshape(len(x), 1)

        else:
            raise ValueError('Please provide one of \'up\', \'down\' or'
                             '\'both\' as a direction.')

        steady_input['values'] = steady_input['values'].tolist()
        if len(params.keys()) == 0:
            params = None

        model = ModelBCMD(
            modelName,
            inputs=steady_input,
            outputs=outputs,
            times=np.arange((len(steady_input['values']))),
            params=params,
            debug=False)

        return model

    def run_steady_state(self):
        """
        Run the steady state model.
        """

        modelName = self.modelName
        params = self.params
        max_val = self.max
        min_val = self.min
        direction = self.direction
        inputs = self.inputs
        outputs = self.outputs

        model = self.make_steady_model(modelName, inputs, outputs, max_val,
                                       min_val, params, direction)

        model.create_initialised_input()
        model.run_from_buffer()
        output = model.output_parse()
        parsed_output = {
            key: list(np.array(val)[99::100])
            for (key, val) in output.items()
        }
        # pprint(output)
        return parsed_output


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
