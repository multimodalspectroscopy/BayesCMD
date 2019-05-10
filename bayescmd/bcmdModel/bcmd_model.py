"""Configure and run a BCMD model.

The BCMD Model class is used to configure and run a BCMD model. It is done by
passing a number of configuration variables, creating a model input file and
then running the model. Models can also be run from a pre-existing input file
and input files can be written to file for later access also.

Attributes
----------
TIMEOUT : :obj:`int`
    Max number of seconds for a simulation to run before being cancelled.
    Default is 30 seconds.
BASEDIR : :obj:`str`
    Path to Base directory, which should be 'BayesCMD'. It is found by running
    the :obj:`bayescmd.util.findBaseDir` method, passing either an environment
    variable or a string to the method.

"""
import pprint
import tempfile
import shutil
import csv
import os
import sys
import datetime

import subprocess
from io import StringIO

import collections
from .input_creation import InputCreator
from bayescmd.util import findBaseDir

# default timeout, in seconds
TIMEOUT = 15
# default base directory - this should be a relative directory path
# leading to BayesCMD/
BASEDIR = os.environ.get('BASEDIR', findBaseDir('BayesCMD'))


class ModelBCMD:
    """Use to create inputs, run simulations and parse outputs.

    Parameters
    ----------
    model_name : :obj:`str`
        Name of model. Should match the modeldef file for model being generated
        i.e. model_name of 'model`' should have a modeldef file
        'model1.modeldef'.
    inputs : :obj:`dict` or :obj:`None`, optional
        Dictionary of model inputs and their values. Has form
        {'names' : :obj:`list` of :obj:`str`,
        'values' : :obj:`list` of :obj:`list` of :obj:`float`}
        where `names` should be a list of each model input name, matching up to
        the model inputs and `values` would be a list of lists, where each
        sublist is the input values for that time point. With this in mind,
        the length of `inputs['values']`` should equal length of `times`.
        Default is None.
    params : :obj:`dict` of :obj:`str`: :obj:`float` or :obj:`None`, optional
        Dictionary of {'parameter': param_value}. Default is None
    times : :obj:`list` of :obj:`float` or :obj:`int` or :obj:`None`, optional
        List of times at which measurement data has been collected and needs
        to be simulated. Default is None.
    outputs : :obj:`list` of :obj:`str` or :obj:`None`, optional
        List of model outputs to return. Default is None.
    burn_in : :obj:`float` or :obj:`int`, optional
        Length of burn in period at start of the simulation. Default is 999
    create_input : :obj:`boolean`, optional
        Boolean indicator as to whether an input file needs creating. Default
        is True.
    input_file : :obj:`str` or :obj:`None`, optional
        Path to existing input file or to where one needs creating. Default is
        None.
    suppress : :obj:`boolean`, optional
        Indicates if console output should be suppressed during model runs.
        This will prevent writing of both stderr and stdout. Default is False.
    workdir : :obj:`str` or :obj:`None`, optional
        Path to working directory if one exists. If not set, it will default
        to a temporary directory in 'tmp/'. If you wish to write input files
        and similar to file, it is recommended that the working directory is
        set manually by the user.
    deleteWorkdir : :obj:`boolean`, optional
        Indicates if the working directory should be deleted after finishing.
        Default is False.
    timeout : :obj:`float` or :obj:`int`, optional
        Maximum length in seconds to let model run before cancelling. Default
        is :obj:`TIMEOUT`.
    basedir : :obj:`str`, optional
        Path to base 'BayesCMD' directory. By default it is set to
        :obj:`BASEDIR`.
    debug : :obj:`boolean`, optional
        Indicates if debugging information should be written to console.
        Default is False.
    testing : :obj:`boolean`, optional
        If True, appends '_test' to coarse and detailed model output. Useful
        if you wish to test settings and want to avoid test results becoming
        mixed in with real result files. Default is False.

    Attributes
    ----------
    model_name : :obj:`str`
        Name of model. Should match the modeldef file for model being generated
        i.e. model_name of 'model`' should have a modeldef file
        'model1.modeldef'.
    inputs : :obj:`dict` or :obj:`None`
        Dictionary of model inputs and their values. Has form
        {'names' : :obj:`list` of :obj:`str`,
        'values' : :obj:`list` of :obj:`list` of :obj:`float`}
        where `names` should be a list of each model input name, matching up to
        the model inputs and `values` would be a list of lists, where each
        sublist is the input values for that time point. With this in mind,
        the length of `inputs['values']`` should equal length of `times`.
        Default is None.
    params : :obj:`dict` of :obj:`str`: :obj:`float` or :obj:`None`
        Dictionary of {'parameter': param_value}. Default is None
    times : :obj:`list` of :obj:`float` or :obj:`int` or :obj:`None`
        List of times at which measurement data has been collected and needs
        to be simulated. Default is None.
    outputs : :obj:`list` of :obj:`str` or :obj:`None`
        List of model outputs to return. Default is None.
    burn_in : :obj:`float` or :obj:`int`
        Length of burn in period at start of the simulation. Default is 999
    create_input : :obj:`boolean`
        Boolean indicator as to whether an input file needs creating. Default
        is True.
    input_file : :obj:`str`
        Path to existing input file or to where one needs creating. If
        :obj:`create_input` is True and no path is given, the input file will
        be written to :obj:`workdir` as :obj:`model_name`.input.
    suppress : :obj:`boolean`
        Indicates if console output should be suppressed during model runs.
        This will prevent writing of both stderr and stdout. Default is False.
    DEVNULL : :obj:`_io.BufferedWriter`
        If :obj:`suppress` is set to True, this will be an io buffer that
        redirects stderr and stdout to the null device.
    workdir : :obj:`str` or :obj:`None`
        Path to working directory if one exists. If not set, it will default
        to a temporary directory in 'tmp/'. If you wish to write input files
        and similar to file, it is recommended that the working directory is
        set manually by the user.

        If no working directory is given, the :obj:`deleteWorkdir` attribute
        will be set to True in order to ensure that the file space does not
        become excessively full during batch runs.
    deleteWorkdir : :obj:`boolean`
        Indicates if the working directory should be deleted after finishing.
        Default is False, but this will always be set to True if :obj:`workdir`
        is set to None.
    timeout : :obj:`float` or :obj:`int`
        Maximum length in seconds to let model run before cancelling. Default
        is :obj:`TIMEOUT`.
    basedir : :obj:`str`
        Path to base 'BayesCMD' directory. By default it is set to
        :obj:`BASEDIR`.
    debug : :obj:`boolean`
        Indicates if debugging information should be written to console.
    program : :obj:`str`
        Path to the compiled model file. This is expected to be in
        ':obj:`basedir`/build', with the name :obj:`model_name`.model.
    output_{coarse,detail} : :obj:`str`
        Location to write coarse and detailed output files to. This will be
        the working directory, with coarse output files having the suffix
        '.out' and detailed output files having the suffix '.detail'.
    output_dict : :obj:`collections.defaultdict(:obj:`list`)`
        Dictionary of output data.

    """

    def __init__(
            self,
            model_name,
            inputs=None,  # Input variables
            params=None,  # Parameters
            times=None,  # Times to run simulation at
            outputs=None,
            burn_in=999,
            create_input=True,
            input_file=None,
            suppress=False,
            workdir=None,  # default is to create a temp directory
            deleteWorkdir=False,
            timeout=TIMEOUT,
            basedir=BASEDIR,
            debug=False,
            testing=False):

        self.model_name = model_name
        self.params = params  # dict of non-default params
        self.inputs = inputs  # any time dependent inputs to the model
        self.times = times
        self.outputs = outputs
        self.burn_in = burn_in

        # Determine if input file is present already or if it needs creating
        self.create_input = create_input

        # Suppression of output files
        self.suppress = suppress
        if suppress:
            self.DEVNULL = open(os.devnull, 'wb')

        # we need working space; we may want to kill it later
        self.deleteWorkdir = deleteWorkdir

        if workdir:
            self.workdir = workdir

            if not os.path.exists(workdir):
                os.makedirs(workdir)
        else:
            self.workdir = tempfile.mkdtemp(prefix=model_name)
            self.deleteWorkdir = True
            if debug:
                print('TEMP DIR: ', self.workdir)

        self.timeout = timeout
        self.debug = debug

        if input_file is not None:
            self.input_file = input_file
        elif create_input:
            self.input_file = os.path.join(self.workdir,
                                           self.model_name + '.input')
        else:
            self.input_file = None

        if testing:
            TEST_PRE = '_test'
        else:
            TEST_PRE = ''

        self.basedir = basedir
        if debug:
            print("BASEDIR set to {}".format(self.basedir))
        self.program = os.path.join(self.basedir, 'build',
                                    self.model_name + '.model')
        self.output_coarse = os.path.join(self.workdir,
                                          self.model_name + TEST_PRE + '.out')
        self.output_detail = os.path.join(
            self.workdir, self.model_name + TEST_PRE + '.detail')
        self.output_dict = collections.defaultdict(list)

    def _cleanupTemp(self):
        """Delete working directory."""
        if self.deleteWorkdir:
            shutil.rmtree(self.workdir)
        return None

    def get_defaults(self):
        """Obtain default model configuration."""
        print('GETTING MODEL DEFAULTS.\n')
        return subprocess.run([self.program, '-s'], stdout=subprocess.PIPE)

    def write_default_input(self):
        """Create and write a default input to file.

        Create an input file using default configuration values and write it to
        the file specified by :obj:`input_file`. All inputs, outputs and
        parameters are set to default values and there is no burn in.
        """
        # Ensure that any existing input files aren't overwritten
        try:
            assert os.path.exists(self.input_file)
            new_input = os.path.splitext(
                self.input_file)[0] + '_{:%H%M%S-%d%m%y}.input'.format(
                    datetime.datetime.now())
            print('Input file %s already exists.\n Renaming as %s' %
                  (self.input_file, new_input))
            input_creator = InputCreator(
                self.times, self.inputs, filename=new_input)
            input_creator.default_creation()
            self.input_file = input_creator.input_file_write()
        except AssertionError:
            input_creator = InputCreator(
                self.times, self.inputs, filename=self.input_file)
            input_creator.default_creation()
            input_creator.input_file_write()

        return True

    def create_default_input(self):
        """Create configured default input file and write to string buffer.

        Using this method allows the configured input file to be written to
        memory, thus reducing the number of operations involving writing to
        disk.
        """
        input_creator = InputCreator(self.times, self.inputs)
        self.input_file = input_creator.default_creation().getvalue()

        return self.input_file

    def write_initialised_input(self):
        """Create and write a custom, initialised input to file.

        Create an input file using configuration values specified by
        :obj:`inputs`, :obj:`times`, :obj:`params` and :obj:`outputs`, and
        write it to the file specified by :obj:`input_file`.
        """
        # Ensure that any existing input files aren't overwritten
        try:
            assert os.path.exists(self.input_file)
            new_input = os.path.splitext(
                self.input_file)[0] + '_{:%H%M%S-%d%m%y}.input'.format(
                    datetime.datetime.now())
            print('Input file %s already exists.\n Renaming as %s' %
                  (self.input_file, new_input))
            input_creator = InputCreator(
                self.times,
                self.inputs,
                params=self.params,
                outputs=self.outputs,
                filename=new_input)
            input_creator.initialised_creation(self.burn_in)
            self.input_file = input_creator.input_file_write()
        except AssertionError:
            input_creator = InputCreator(
                self.times,
                self.inputs,
                params=self.params,
                outputs=self.outputs,
                filename=self.input_file)
            input_creator.initialised_creation(self.burn_in)
            input_creator.input_file_write()

        return True

    def create_initialised_input(self):
        """Create a custom, initialised input and write to buffer.

        Create an input file using configuration values specified by
        :obj:`inputs`, :obj:`times`, :obj:`params` and :obj:`outputs`, and
        write it to the file specified by :obj:`input_file`.

        The configured input file will be written to memory, thus reducing the
        number of operations involving writing to disk.
        """
        input_creator = InputCreator(
            self.times, self.inputs, params=self.params, outputs=self.outputs)
        f_out = input_creator.initialised_creation(self.burn_in)

        if self.debug:
            print(f_out.getvalue(), file=sys.stdout)
            f_out.seek(0)

        self.input_file = f_out.getvalue()
        return self.input_file

    def run_from_file(self):
        """Run model using an input file found at :obj:`input_file`.

        The model will be run using an already created input file that has been
        written manually or using the :obj:`write_default_input` or
        :obj:`write_initialised_input` methods.
        """
        try:
            assert os.path.exists(self.input_file)
            if self.debug:
                print("\n\nOutput goes to:\n\tCOARSE:%s\n\tDETAIL:%s\n\n" %
                      (self.output_coarse, self.output_detail))
            if self.suppress:
                # invoke the model program as a subprocess
                succ = subprocess.run(
                    [
                        self.program, '-i', self.input_file, '-o',
                        self.output_coarse, '-d', self.output_detail
                    ],
                    stdout=self.DEVNULL,
                    stderr=self.DEVNULL,
                    timeout=self.timeout,
                    check=True)
            else:
                stdoutname = os.path.join(self.workdir,
                                          '%s.stdout' % (self.model_name))
                stderrname = os.path.join(self.workdir,
                                          '%s.stderr' % (self.model_name))

                # if opening these files fails, we may be in trouble anyway
                # but don't peg out just because of this -- let the the failure
                # happen somewhere more important
                try:
                    f_out = open(stdoutname, 'w')
                except IOError:
                    f_out = None

                try:
                    f_err = open(stderrname, 'w')
                except IOError:
                    f_err = None

                # invoke the model program as a subprocess
                succ = subprocess.run(
                    [
                        self.program, '-i', self.input_file, '-o',
                        self.output_coarse, '-d', self.output_detail
                    ],
                    stdout=f_out,
                    stderr=f_err,
                    timeout=self.timeout,
                    check=True)

                if f_out:
                    f_out.close()
                if f_err:
                    f_err.close()
        except AssertionError:
            print("Input file doesn't exist. Can't run from file.")

        return None

    def run_from_buffer(self):
        """Run the model using an input file written to memory.

        The input file will need to have been created using
        :obj:`create_initialised_input` or :obj:`create_default_input`.
        """
        # Ensure that input file has seeked to 0
        if self.debug:
            print("Output goes to:\n\tCOARSE:%s\n\tDETAIL:%s" %
                  (self.output_coarse, self.output_detail))
        if self.suppress:
            # invoke the model program as a subprocess
            result = subprocess.run(
                [self.program, '-I'],
                input=self.input_file.encode(),
                stdout=subprocess.PIPE,
                stderr=self.DEVNULL,
                timeout=self.timeout,
                check=True)
        else:
            stderrname = os.path.join(self.workdir, '%s.stderr' %
                                      ("buffer_" + self.model_name))

            # if opening these files fails, we may be in trouble anyway
            # but don't peg out just because of this -- let the the failure
            # happen somewhere more important

            try:
                f_err = open(stderrname, 'w')
            except IOError:
                f_err = None

            # invoke the model program as a subprocess
            result = subprocess.run(
                [self.program, '-I', '-d', self.output_detail],
                input=self.input_file.encode(),
                stdout=subprocess.PIPE,
                stderr=f_err,
                timeout=self.timeout,
                check=True)

            if f_err:
                f_err.close()

        self.output_coarse = StringIO(result.stdout.decode())

        if self.debug:
            print(self.output_coarse.getvalue(), file=sys.stderr)
            self.output_coarse.seek(0)
        return result

    def output_parse(self):
        """Parse the output files into a dictionary.

        This allows the model output to be sent to JSON, if using WeBCMD, or
        simply processed in a more pythonic fashion for any Bayesian (or
        similar) analysis.
        """
        # Check if file is open

        try:
            file_out = open(self.output_coarse)
        except TypeError:
            file_out = self.output_coarse

        if self.debug:
            pprint.pprint(file_out.read(), stream=sys.stderr)
            file_out.seek(0)

        for d in csv.DictReader(file_out, delimiter='\t'):
            for key, value in d.items():
                if key == 'ERR':
                    pass
                else:
                    try:
                        self.output_dict[key].append(float(value))
                    except (ValueError, TypeError) as e:
                        self.output_dict[key].append('NaN')

        self._cleanupTemp()
        return self.output_dict
