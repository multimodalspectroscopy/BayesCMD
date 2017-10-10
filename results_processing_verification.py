"""Process results from 280917."""
import os
import argparse
from bayescmd.results_handling import kde_plot
from bayescmd.results_handling import data_import
from bayescmd.results_handling import plot_repeated_outputs
from bayescmd.results_handling import histogram_plot
from bayescmd.util import findBaseDir
import numpy as np
import matplotlib.pyplot as plt
BASEDIR = os.path.abspath(findBaseDir('BayesCMD'))

ap = argparse.ArgumentParser('Choose results to process:')
ap.add_argument(
    'input_file',
    metavar="INPUT_FILE",
    help='File containing parameters from multiple runs')

args = ap.parse_args()

pfile = args.input_file  # 'concatenated_results_190917.csv'
params = {
    r'r_t': (0.009, 0.027),
    r'r_0': (0.0063, 0.0189),
    r'r_m': (0.0135, 0.0405)
}

input_path = os.path.join(BASEDIR, 'data', 'bayes-test-data.csv')

targets = ['CCO']
model_name = 'BS'
inputs = ['u']

config = {
    "model_name": model_name,
    "targets": targets,
    "inputs": inputs,
    "parameters": params,
    "input_path": input_path,
    "zero_flag": {
        "CCO": False
    }
}

results = data_import(pfile)

figPath = "/home/buck06191/Dropbox/phd/BayesCMD/Figures/"
print("Plotting total histogram")
hist1 = histogram_plot(results)
hist1.savefig(
    os.path.join(figPath, 'full_histogram_test.png'), bbox_inches='tight')
print("Plotting fraction histogram")
hist2 = histogram_plot(results, fraction=0.01)
hist2.savefig(
    os.path.join(figPath, 'fraction_histogram_test.png'), bbox_inches='tight')
for f in [0.01, 0.1, 1.0]:
    print("Considering lowest {}% of values".format(f))
    # print("Generating scatter plot")
    # scatter_dist_plot(results, params, f, n_ticks=4)
    # plt.show()
    print("Generating KDE plot")
    g = kde_plot(results, params, f, n_ticks=4)
    g.fig.savefig(
        os.path.join(figPath, 'kde_{}_test.png'
                     .format(str(f).replace('.', '_'))),
        bbox_inches='tight')
    print("Generating averaged time series plot")
    fig = plot_repeated_outputs(results, n_repeats=25, frac=f, **config)
    fig.set_size_inches(18.5, 12.5)
    fig.savefig(
        os.path.join(figPath, 'TS_{}_test.png'
                     .format(str(f).replace('.', '_'))),
        dpi=100)
