"""Process results from 280917."""
import os
import argparse
from bayescmd.results_handling import kde_plot
from bayescmd.results_handling import scatter_dist_plot
from bayescmd.results_handling import data_import
from bayescmd.results_handling import plot_repeated_outputs
from bayescmd.results_handling import histogram_plot
from bayescmd.results_handling import data_merge_by_date
from bayescmd.abc import import_actual_data
from bayescmd.abc import priors_creator
from bayescmd.util import findBaseDir
from distutils import dir_util
import json
import numpy as np

BASEDIR = os.path.abspath(findBaseDir('BayesCMD'))

ap = argparse.ArgumentParser('Choose results to process:')
ap.add_argument(
    'parent_dir',
    metavar="PARENT_DIR",
    help='Parent directory holding model run folders')

ap.add_argument(
    'conf',
    metavar="config_file",
    help='Config file used to generate model runs')

args = ap.parse_args()

######### LOAD DATA ############

# Define the location of the parameter combinations and distances -
#       use `data_merge_by_date` and `data_merge_by_batch` to help with this

date = '280917'
pfile = data_merge_by_date(data, args.parent_directory)


results = data_import(pfile)


# Set input file name here for reading - this is typically the measured data
input_file_name = ""

input_path = os.path.join(BASEDIR, , input_file_name)
d0 = import_actual_data(input_path)

# Add paths to any other required files e.g. OpenOpt fits
openopt_path = os.path.join(BASEDIR, 'data', 'model_run_output.csv')

######### CONFIG  FILE ###########
# Extract information from config.

with open(args.conf, 'r') as conf_f:
    conf = json.load(conf_f)


targets = conf['targets']
model_name = conf['model_name']
inputs = conf['inputs']

# Define priors based on config options.
params = priors_creator(conf['priors']['defaults'],
                        conf['priors']['variation'])

config = {
    "model_name": model_name,
    "targets": targets,
    "inputs": inputs,
    "parameters": params,
    "input_path": input_path,
    "openopt_path": openopt_path,
    "zero_flag": {k: False for k in targets}
}






d = 'euclidean'
lim = 1000

## Set path to save figures to
figPath = "/home/buck06191/Dropbox/phd/Bayesian_fitting/{}/{}/"\
    "Figures/{}".format(model_name, date, d)
dir_util.mkpath(figPath)

## Plot histograms of errors for checking 
print("Plotting total histogram")
hist1 = histogram_plot(results, frac=1)
hist1.savefig(
    os.path.join(figPath, 'full_histogram_real.png'), bbox_inches='tight')
print("Plotting fraction histogram")
hist2 = histogram_plot(results, limit=lim)
hist2.savefig(
    os.path.join(figPath, 'fraction_histogram_real.png'), bbox_inches='tight')

## Print posteriors
print("Considering lowest {} values".format(lim))

print("Generating KDE plot")
openopt_medians = {"r_t": 0.016243,
                   "sigma_coll": 78.000000,
                   "cytox_tot_tis": 0.006449,
                   "Vol_mit": 0.084000,
                   "O2_n": 0.030000,
                   "r_0": 0.011808,
                   "v_cn": 30.000000,
                   "r_m": 0.021274}
g = kde_plot(results, params, limit=lim, n_ticks=4,
             openopt_medians=openopt_medians)
g.fig.savefig(
    os.path.join(figPath, 'kde_{}_real.png'
                 .format(str(lim).replace('.', '_'))),
    bbox_inches='tight')

## Print posterior predictive
print("Generating averaged time series plot")
fig = plot_repeated_outputs(results, n_repeats=25, limit=lim, **config)
fig.set_size_inches(18.5, 12.5)
fig.savefig(
    os.path.join(figPath, 'TS_{}_real.png'
                 .format(str(lim).replace('.', '_'))),
    dpi=100)
