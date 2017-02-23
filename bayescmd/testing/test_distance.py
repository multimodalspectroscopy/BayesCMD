from ..bcmdModel.bcmd_model import ModelBCMD
from nose.tools import assert_true, assert_equal, with_setup
import numpy.testing as np_test
import filecmp
import os

BASEDIR = os.path.abspath(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))))
assert os.path.basename(BASEDIR) == 'BayesCMD'
print(BASEDIR)


def test_output_dict_from_buffer():
    """
    Check that an output file will be read and processed correctly.
    """

    times = list(range(0, 30, 5))
    inputs = {"names": ['V'],
              "values": [
                  [1],
                  [0],
                  [-1],
                  [0],
                  [1]
    ]
    }
    params = None

    default_model = ModelBCMD('rc',
                              inputs,
                              params,
                              times,
                              input_file=None,
                              create_input=True,
                              testing=True,
                              workdir=os.path.join('.', 'test_files'),
                              debug=False,
                              basedir=BASEDIR)

    default_model.create_default_input()
    default_model.run_from_buffer()
    vc_test = default_model.output_parse()['Vc']
    np_test.assert_almost_equal(vc_test, [0.99326201853524232,
                                          0.0066925479138209504,
                                          -0.99321690724584777,
                                          -0.0066922439555526904,
                                          0.99321692632751313],
                                err_msg='Vc Output not the same')
    os.remove(default_model.output_detail)
