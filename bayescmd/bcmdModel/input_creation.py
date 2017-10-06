"""Create input files for use with a BCMD model.

Input files are needed in order to set model parameters and provide driving
inputs.
"""
import os
import sys
from io import StringIO


class InputCreator:
    """Create an input file by passing relevant information to the class.

    This input file is then used to create an input file that can either be
    written to file or kept in buffer and passed directly to the model.

    Parameters
    ----------
    times : :obj:`list` of :obj:`float` or :obj:`int`
        List of times at which measurement data has been collected and needs
        to be simulated.
    inputs : dict
        Dictionary of model inputs and their values. Has form
        {'names' : :obj:`list` of :obj:`str`,
        'values' : :obj:`list` of :obj:`list` of :obj:`float`}
        where `names` should be a list of each model input name, matching up to
        the model inputs and `values` would be a list of lsits, where each
        sublist is the input values for that time point. With this in mind,
        the length of `inputs['values']`` should equal length of `times`.
    filename : :obj:`str`, optional
        Name of the input file to be written to if writing to file is required.
        Default is :obj:`None`.
    params : :obj:`dict` of :obj:`str`: :obj:`float`, optional
        Dictionary of {'parameter': param_value}
    outputs : :obj:`list` of :obj:`str`, optional
        List of model outputs to return.

    Attributes
    ----------
    times : :obj:`list` of :obj:`float` or :obj:`int`
        List of times at which measurement data has been collected and needs
        to be simulated.
    inputs : dict
        Dictionary of model inputs and their values. Has form
        {'names' : :obj:`list` of :obj:`str`,
        'values' : :obj:`list` of :obj:`list` of :obj:`float`}
        where `names` should be a list of each model input name, matching up to
        the model inputs and `values` would be a list of lsits, where each
        sublist is the input values for that time point. With this in mind,
        the length of `inputs['values']`` should equal length of `times`.
    f_out : :obj:`StringIO()`
        String buffer object to which the input file will be written.
    filename : str
        Name of the input file to be written to if writing to file is required.
        Default is :obj:`None`.
    params : dict of :obj:`str`: :obj:`float`.
        Dictionary of {'parameter': param_value}
    outputs : :obj:`list` of :obj:`str`
        List of model outputs to return.

    """

    def __init__(self, times, inputs, outputs=None, params=None,
                 filename=None):
        self.times = times
        self.inputs = inputs
        self.filename = filename
        self.params = params
        self.outputs = outputs
        self.f_out = StringIO()

    def input_file_write(self):
        """Write input file from buffer to file."""
        with open(self.filename, 'w') as f:
            f.write(self.f_out.getvalue())
        print("Input file written to %s" % os.path.abspath(self.filename))
        return self.filename

    def default_creation(self):
        """Create a default input file from given arguments.

        Assumes parameters remain unchanged from default values.

        Returns
        -------
        :obj:`String.IO()`
            Returns the input file as a String.IO() buffer object.

        """
        if self.inputs is not None:
            assert len(self.times) == len(self.inputs['values']), "Different" \
                "number of time steps in log and in data:\n \t" \
                "time steps = %d \n\t" \
                "input steps = %d" % (len(self.times[:-1]),
                                      len(self.inputs['values']))

        self.f_out.write('# File created using BayesCMD file creation\n')
        self.f_out.write('@%d\n' % len(self.times))
        if self.inputs is not None:
            self.f_out.write(': %d ' % len(self.inputs['names']) +
                             ' '.join(self.inputs['names']) + '\n')
            self.f_out.write('= -1 ' + str(self.times[0]) + ' ' + ' '.join(
                str(v) for v in self.inputs['values'][0]) + '\n')
            for ii in range(len(self.times[:-1])):
                self.f_out.write(
                    '= %f %f ' % (self.times[ii], self.times[ii + 1]) +
                    ' '.join(str(v) for v in self.inputs['values'][ii]) + '\n')
        else:
            self.f_out.write(':0\n= -1 ' + str(self.times[0]) + '\n')
            for ii in range(len(self.times[:-1])):
                self.f_out.write('= %f %f ' % (self.times[ii],
                                               self.times[ii + 1]) + '\n')

        self.f_out.seek(0)
        return self.f_out

    def initialised_creation(self, burn_in):
        """Create an input file from given arguments.

        Creates an input file thatcan have non-default parameter values and
        outputs, as well as a burn in period. Assumes parameters remain
        constant for the full duration of the simulation.

        Parameters
        ----------
        burn_in : :obj:`float` or :obj:`int`
            Length of burn in period at start of the simulation.

        Returns
        -------
        :obj:`String.IO()`
            Returns the input file as a String.IO() buffer object.
        """
        self.f_out.write('# File created using BayesCMD file creation\n')

        # Create lists for initialised names and values
        init_names = []
        init_vals = []
        if self.params is not None:
            for k, v in self.params.items():
                init_names.append(k)
                init_vals.append(v)

        if self.inputs is not None:
            init_names.extend(self.inputs['names'])
            init_vals.extend(self.inputs['values'][0])

        if burn_in > 0:
            self.f_out.write('@ %d\n' % (len(self.times) + 1))
            self.f_out.write('>>> 0\n!0\n')
            self.f_out.write(':%d ' % len(init_names) + ' '.join(init_names) +
                             '\n')

            self.f_out.write('= -%f -1 ' % (burn_in + 1) + ' '.join(
                str(v) for v in init_vals) + '\n')
        else:
            self.f_out.write('@ %d\n' % (len(self.times) + 1))
            self.f_out.write('>>> 0\n!0\n')
            self.f_out.write(':%d ' % len(init_names) + ' '.join(init_names) +
                             '\n')

            self.f_out.write('= -2 -1 ' + ' '.join(str(v)
                                                   for v in init_vals) + '\n')
        # Set post burn in outputs and values
        if len(self.outputs) == 0:
            self.f_out.write('>>> 1 t \n!!!\n')
        elif len(self.outputs) == 1:
            self.f_out.write('>>> 2 t %s\n!!!\n' % (self.outputs[0]))
        else:
            if self.inputs is not None:
                self.f_out.write('>>> %d t ' % (len(self.outputs) + len(
                    self.inputs['names']) + 1, ) + ' '.join(self.outputs) + ' '
                                 + ' '.join(self.inputs['names']) + '\n!!!\n')
            else:
                self.f_out.write('>>> %d t ' % (len(self.outputs) + 1, ) +
                                 ' '.join(self.outputs) + ' ' + '\n!!!\n')

        if self.inputs is not None:
            assert len(self.times) == len(self.inputs['values']), "Different "\
                "number of time steps in log and in data:\n \t" \
                "time steps = %d \n\t" \
                "input steps = %d" % (len(self.times),
                                      len(self.inputs['values']))
            self.f_out.write(':%d ' % len(self.inputs['names']) +
                             ' '.join(self.inputs['names']) + '\n')
            self.f_out.write('= -1 %s ' % (self.times[0]) + ' '.join(
                str(v) for v in self.inputs['values'][0]) + '\n')
            for ii in range(len(self.times[:-1])):
                self.f_out.write('= %f %f ' % (
                    self.times[ii], self.times[ii + 1]) + ' '.join(
                        str(v) for v in self.inputs['values'][ii + 1]) + '\n')
        else:
            self.f_out.write(':0\n= -1 %f\n' % (self.times[0]))
            for ii in range(len(self.times[:-1])):
                self.f_out.write('= %f %f ' % (self.times[ii],
                                               self.times[ii + 1]) + '\n')

        self.f_out.seek(0)
        return self.f_out


#  TODO: Create method to parse a data file into the inputs dict form
if __name__ == '__main__':
    try:
        output = os.path.join('.', 'test_files', 'output_input.txt')
        values = [[1 * i, 2 * i, 3 * i] for i in range(1, 11)]
        inputs = {"names": ['x', 'y', 'z'], "values": values}
        times = list(range(11))
        input_creator = InputCreator(output, times, inputs)
        input_creator.default_creation()

        os.remove(output)
        input_creator_2 = InputCreator(output, times, inputs)
        output = input_creator_2.default_creation_2()
        print(output.getvalue())
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
