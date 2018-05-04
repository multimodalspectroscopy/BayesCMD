"""Process results from 230218."""
import os
import argparse
from bayescmd.results_handling import kde_plot
from bayescmd.results_handling import scatter_dist_plot
from bayescmd.results_handling import data_import
from bayescmd.results_handling import plot_repeated_outputs
from bayescmd.results_handling import histogram_plot
from bayescmd.results_handling import data_merge
from bayescmd.abc import priors_creator
from bayescmd.util import findBaseDir
import json

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


pfile = data_merge('230218', args.parent_dir)

with open(args.conf, 'r') as conf_f:
    conf = json.load(conf_f)
params = priors_creator(conf['priors']['defaults'],
                        conf['priors']['variation'])

input_path = os.path.join(BASEDIR, 'data', 'SA_clean_cropped.csv')


targets = conf['targets']
model_name = conf['model_name']
inputs = conf['inputs']

config = {
    "model_name": model_name,
    "targets": targets,
    "inputs": inputs,
    "parameters": params,
    "input_path": input_path,
    "zero_flag": {k: False for k in targets}
}

results = data_import(pfile)

figPath = "/home/buck06191/Dropbox/phd/hypothermia/Figures/Bayesian_fitting/"
print("Plotting total histogram")
hist1 = histogram_plot(results)
hist1.savefig(
    os.path.join(figPath, 'full_histogram_real.png'), bbox_inches='tight')
print("Plotting fraction histogram")
hist2 = histogram_plot(results, fraction=0.01)
hist2.savefig(
    os.path.join(figPath, 'fraction_histogram_real.png'), bbox_inches='tight')
for f in [1.0]:
    print("Considering lowest {}% of values".format(f))
    print("Generating scatter plot")
    scatter_dist_plot(results, params, f, n_ticks=4)
    print("Generating KDE plot")
    g = kde_plot(results, params, f, n_ticks=4,
                 median_file=os.path.join(figPath, "medians.txt"))
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
