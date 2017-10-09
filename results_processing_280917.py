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
    r'r_t': (0.0135, 0.0225),
    r'r_0': (0.00945, 0.01575),
    r'r_m': (0.02025, 0.03375),
    r'cytox_tot_tis': (0.004125, 0.006875),
    r'Vol_mit': (0.05025, 0.08375),
    r'O2_n': (0.018, 0.03),
    r'v_cn': (30.0, 50.0),
    r'sigma_coll': (47.0925, 78.4875)
}

input_path = os.path.join(BASEDIR, 'data', 'hx01.csv')
openopt_path = os.path.join(BASEDIR, 'data', 'model_run_output.csv')

targets = ['Vmca', 'CCO']
model_name = 'BS'
inputs = ['Pa_CO2', 'P_a', 'SaO2sup']

config = {
    "model_name": model_name,
    "targets": targets,
    "inputs": inputs,
    "parameters": params,
    "openopt_path": openopt_path,
    "input_path": input_path
}

results = data_import(pfile)

figPath = "/home/buck06191/Dropbox/phd/BayesCMD/Figures/"
print("Plotting total histogram")
hist1 = histogram_plot(results)
hist1.savefig(
    os.path.join(figPath, 'full_histogram_real.png'), bbox_inches='tight')
print("Plotting fraction histogram")
hist2 = histogram_plot(results, fraction=0.01)
hist2.savefig(
    os.path.join(figPath, 'fraction_histogram_real.png'), bbox_inches='tight')
for f in [0.01, 0.1, 1.0]:
    print("Considering lowest {}% of values".format(f))
    # print("Generating scatter plot")
    # scatter_dist_plot(results, params, f, n_ticks=4)
    # plt.show()
    print("Generating KDE plot")
    g = kde_plot(results, params, f, n_ticks=4)
    g.fig.savefig(
        os.path.join(figPath, 'kde_{}_real.png'
                     .format(str(f).replace('.', '_'))),
        bbox_inches='tight')
    print("Generating averaged time series plot")
    fig = plot_repeated_outputs(results, n_repeats=25, frac=f, **config)
    fig.set_size_inches(18.5, 12.5)
    fig.savefig(
        os.path.join(figPath, 'TS_{}_real.png'
                     .format(str(f).replace('.', '_'))),
        dpi=100)

# TODO: Fix issue with plot formatting, cutting off axes etc
# TODO: Fix issue with time series cutting short.
rt_range = np.linspace(0.01, 0.03, 100)
rm_range = np.linspace(0.01, 0.04, 100)
r0_range = np.linspace(0.007, 0.0175, 100)
volmit_range = np.linspace(0.02, 0.12, 100)
cytox_range = np.linspace(0.0025, 0.009, 100)

ranges = [rt_range, rm_range, r0_range, volmit_range, cytox_range]
