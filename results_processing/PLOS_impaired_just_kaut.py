"""Process results from 230218."""
import os
import argparse
import sys
sys.path.append('..')
from bayescmd.results_handling import kde_plot
from bayescmd.results_handling import scatter_dist_plot
from bayescmd.results_handling import data_import
from bayescmd.results_handling import plot_repeated_outputs
from bayescmd.results_handling import histogram_plot
from bayescmd.results_handling import data_merge_by_batch
from bayescmd.abc import import_actual_data
from bayescmd.abc import priors_creator
from bayescmd.util import findBaseDir
import json
from distutils import dir_util

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

pfile = data_merge_by_batch(args.parent_dir)
# pfile = os.path.abspath(os.path.join(args.parent_dir, 'all_parameters.csv'))

with open(args.conf, 'r') as conf_f:
    conf = json.load(conf_f)


params = conf['priors']

input_path = os.path.join(BASEDIR,
                          'PLOS_paper',
                          'data',
                          'simulated_smooth_combined_impaired_ABP.csv')

d0 = import_actual_data(input_path)

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
print(results.columns)


# Set accepted limit, lim
lim = 1000
distances = ['{}_euclidean'.format(t) for t in config['targets']]
distances.append('euclidean')
for d in distances:
    print("Working on {}".format(d.upper()))
    figPath = "/home/buck06191/Dropbox/phd/Bayesian_fitting/{}/{}/{}/{}/"\
        "Figures/{}".format(model_name, 'PLOS_paper',
                            'just_kaut', 'Impaired', d)

    dir_util.mkpath(figPath)
    print("Plotting total histogram")
    hist1 = histogram_plot(results, distance=d, frac=1)
    hist1.savefig(
        os.path.join(figPath, 'full_histogram_real.png'), bbox_inches='tight')
    print("Plotting fraction histogram")
    hist2 = histogram_plot(results, distance=d, limit=lim)
    hist2.savefig(
        os.path.join(figPath, 'fraction_histogram_real.png'),
        bbox_inches='tight')
    print("Considering lowest {} values".format(lim))
    # print("Generating scatter plot")
    # scatter_dist_plot(results, params, limit=lim, n_ticks=4)
    print("Generating KDE plot")
    g = kde_plot(results, params, limit=lim, n_ticks=4, d=d,
                 median_file=os.path.join(figPath, "medians.txt"))
    g.fig.savefig(
        os.path.join(figPath, 'kde_{}_real.png'
                     .format(str(lim).replace('.', '_'))),
        bbox_inches='tight')
    print("Generating averaged time series plot")
    fig = plot_repeated_outputs(results, n_repeats=25, limit=lim,
                                      distance=d, **config)
    fig.set_size_inches(18.5, 12.5)
    fig.savefig(
        os.path.join(figPath, 'TS_{}_real.png'
                     .format(str(lim).replace('.', '_'))),
        dpi=100)

# TODO: Fix issue with plot formatting, cutting off axes etc
# TODO: Fix issue with time series cutting short.
