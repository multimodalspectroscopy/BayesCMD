import os
import sys
from io import StringIO


class InputCreator:

    def __init__(self, times, inputs, filename=None):
        self.filename = filename
        self.times = times
        self.inputs = inputs
        self.f_out = StringIO()

    def input_file_write(self):
        with open(self.filename, 'w') as f:
            f.write(self.f_out.getvalue())
        print("Input file written to %s" % os.path.abspath(self.filename))
        return self.filename

    def default_creation(self):
        """
        Create a default input file from given arguments. Assumes parameters
        remain unchanged.
        :param times: List of times, as per input data. Length should be 1 more
        than actual number of time steps (0-base)
        :param inputs: dictionary of 'inputs'. This is any thing, inputs or
        params, which are defined at subsequent timepoints.
        :return: Output string buffer
        """
        assert len(self.times[:-1]) == len(self.inputs['values']), "Different" \
            "number of time steps in log and in data:\n \t" \
            "time steps = %d \n\t" \
            "param steps = %d" % (len(self.times)[:-1],
                                  len(self.inputs['values']))

        self.f_out.write('# File created using BayesCMD file creation\n')
        self.f_out.write('@%d\n' % len(self.times[:-1]))
        self.f_out.write(': %d ' % len(self.inputs['names']) +
                         ' '.join(self.inputs['names']) + '\n')
        for ii, time in enumerate(self.times[:-1]):
            self.f_out.write('= %d %d ' % (self.times[ii],
                                           self.times[ii + 1]) +
                             ' '.join(str(v) for v in self.inputs['values'][ii]) + '\n')

        self.f_out.seek(0)
        return self.f_out

    def initialised_creation(self):
        """
        Create an input file from given arguments that will iinitialise.
        Assumes parameters remain unchanged.
        :param times: List of times, as per input data. Length should be 1 more
        than actual number of time steps (0-base)
        :param inputs: dictionary of 'inputs'. This is any thing, inputs or
        params, which are defined at subsequent timepoints.
        :return: Output string buffer
        """
        assert len(self.times[:-1]) == len(self.inputs['values']), "Different" \
            "number of time steps in log and in data:\n \t" \
            "time steps = %d \n\t" \
            "param steps = %d" % (len(self.times)[:-1],
                                  len(self.inputs['values']))

        self.f_out.write('# File created using BayesCMD file creation\n')
        self.f_out.write('@%d\n' % len(self.times[:-1]))
        self.f_out.write(': %d ' % len(self.inputs['names']) +
                         ' '.join(self.inputs['names']) + '\n')
        for ii, time in enumerate(self.times[:-1]):
            self.f_out.write('= %d %d ' % (self.times[ii],
                                           self.times[ii + 1]) +
                             ' '.join(str(v) for v in self.inputs['values'][ii]) + '\n')

        self.f_out.seek(0)
        return self.f_out

if __name__ == '__main__':
    try:
        output = os.path.join('.', 'test_files', 'output_input.txt')
        values = [[1 * i, 2 * i, 3 * i] for i in range(1, 11)]
        inputs = {"names": ['x', 'y', 'z'],
                  "values": values}
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
