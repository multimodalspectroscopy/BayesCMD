import sys
sys.path.append('.')
import os
from bayescmd.util import findBaseDir
import matplotlib.pyplot as plt
import numpy as np


BASEDIR = findBaseDir()
assert os.path.basename(os.path.abspath(BASEDIR)) == 'BayesCMD'


from bayescmd.bcmdModel import ModelBCMD


timed_model = ModelBCMD('lotka-volterra',
                        inputs= None,  # Input variables
                        params= {'a': 0.1,
                                 'b': 0.02,
                                 'c': 0.02,
                                 'd': 0.4,
                                 'x': 10,
                                 'y': 10},  # Parameters
                        times= list(range(1001)),  # Times to run simulation at
                        outputs=['x','y'],
                        debug=False)


def run_model(model):
    model.create_initialised_input()
    model.run_from_buffer()
    output = model.output_parse()
    return output

def sample_data(data, n_samples=25):
    sample_points = np.random.randint(0,len(data['t']), n_samples)
    return np.array(list(data.values())).transpose()[sample_points,:]


if __name__ == '__main__':
    LV_test = run_model(timed_model)
    data = sample_data(LV_test)


    plt.plot(LV_test['t'],LV_test['x'])
    plt.plot(LV_test['t'],LV_test['y'])
    plt.show()