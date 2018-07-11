"""Generate simulated data for PLOS paper."""
import sys
import os
import csv
import numpy as np
sys.path.append('..')
from bayescmd import signal_generator as sg
import matplotlib.pyplot as plt

sine = {'kind': 'sine', 'lo': -10, 'hi': 10, "freq": 0.00027, "phi": np.pi}
walk = {'kind': 'walk', 'sd': 0.1, 'lo': None}
noise = {'kind': 'gaussian', 'sd': 0.1, 'lo': None}
square = {'kind': 'square', 'lo': -5, 'hi': 5, 'freq': 0.0005, "phi": 100}
step = {"kind": "tophat", "hi": 5}

N = 2000

# wv = sg.generate(n=N, timescale=1, specs=[sine, square, noise, walk],
#                  lo=75, hi=105)
# wv = sg.generate(n=N, timescale=1, specs=[step, noise], lo=75, hi=105)

wv = sg.generate(n=N, timescale=1, specs=[sine, noise, walk],
                 lo=0.6, hi=0.98)

plt.plot(wv['t'], wv['signal'])
plt.show()


save_data = str(input("Do you want to save the data: y/N?") or "N")
fname = 'simulated_hypoxia.csv'
basedir = os.path.dirname(os.path.abspath(__file__))
print(basedir)
if save_data == "y":
    fname = str(input("filename: (Default: %s)\n%s/" % (fname, basedir))
                or fname)
    with open(os.path.join(basedir, fname), 'w') as wf:
        w = csv.writer(wf)
        w.writerow(('t', 'SaO2sup'))
        for i in range(N):
            w.writerow((wv['t'][i], wv['signal'][i]))
    print("Data saved in %s" % (os.path.join(basedir, fname)))
elif save_data == "N" or save_data == "n":
    print("Data not saved.")
else:
    print("Argument not recognised.")
