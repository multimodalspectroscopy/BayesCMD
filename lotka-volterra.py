import sys
sys.path.append('.')
import os
from bayescmd.util import findBaseDir
import matplotlib.pyplot as plt
import numpy as np
import random


BASEDIR = findBaseDir()
assert os.path.basename(os.path.abspath(BASEDIR)) == 'BayesCMD'


from bayescmd.bcmdModel import ModelBCMD


timed_model = ModelBCMD('lotka-volterra',
                        inputs= None,  # Input variables
                        params= {'a': 1,
                                 'b': 1,
                                 'c': 1,
                                 'd': 1,
                                 'x': 1,
                                 'y': 0.5},  # Parameters
                        times= list(range(16)),  # Times to run simulation at
                        outputs=['x','y'],
                        debug=False)


def run_model(model):
    model.create_initialised_input()
    model.run_from_buffer()
    output = model.output_parse()
    return output

def sample_data(data, noisy_data, n_samples=25):
    sample_points = random.sample(range(len(data['t'])), n_samples)
    return {k : np.array(v)[sample_points] for k, v in data.items()},\
           {k: np.array(v)[sample_points] for k, v in noisy_data.items()},\
           sample_points


def add_noise(data):
    d = {k: np.array(v)+np.random.normal(0, 0.5, len(v)) for k,v in data.items() if k != 't'}
    d['t']=data['t']
    return d

if __name__ == '__main__':
    data = run_model(timed_model)
    noisy_data = add_noise(data)
    sample, noisy_sample, sample_points = sample_data(data, noisy_data, 8)


    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)

    ax.scatter(sample['t'], sample['x'], c='red', marker='+' )
    plt.scatter(sample['t'], sample['y'], c='red', marker='^')
    ax.scatter(noisy_sample['t'], noisy_sample['x'], c='blue', marker='+')
    plt.scatter(noisy_sample['t'], noisy_sample['y'], c='blue', marker='^')
    ax.plot(data['t'], data['x'], 'k-')
    ax.plot(data['t'], data['y'], 'k--')
    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()
    ax2 = fig.add_subplot(2, 1, 2)
    noise = sample['x']-noisy_sample['x']
    ax2.plot(sample['t'],noisy_sample['t'],'.')
    ax2.set_xlim((0,16))
    plt.show()

