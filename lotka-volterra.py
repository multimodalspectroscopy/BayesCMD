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
    return {k : np.array(v)[sample_points] for k, v in data.items()}

def add_noise(data):
    d = {k: np.array(v)+np.random.normal(0, 0.5, len(v)) for k,v in data.items() if k != 't'}
    d['t']=data['t']
    return d

if __name__ == '__main__':
    data = run_model(timed_model)
    noisy_data = add_noise(data)
    sample = sample_data(data)
    noisy_sample = sample_data(noisy_data)


    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.scatter(sample['t'], sample['x'], c='red', marker='+' )
    #plt.scatter(sample['t'], sample['y'], c='red', marker='*')
    ax.scatter(noisy_sample['t'], noisy_sample['x'], c='blue', marker='+')
    #plt.scatter(noisy_sample['t'], noisy_sample['y'], c='blue', marker='*')
    ax2 = fig.add_subplot(2, 1, 2)
    noise = sample['x']-noisy_sample['x']
    ax2.plot(data['t'],noisy_data['t'],'.')
    plt.show()