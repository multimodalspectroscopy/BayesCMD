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
import matplotlib.pyplot as plt
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

# pfile = data_merge_by_batch(args.parent_dir)
pfile = os.path.abspath(os.path.join(args.parent_dir, 'posterior_parameters.csv'))# 'reduced_sorted_parameters.csv'))

with open(args.conf, 'r') as conf_f:
    conf = json.load(conf_f)
# params = priors_creator(conf['priors']['defaults'],
#                         conf['priors']['variation'])
params = conf['priors']

input_path = os.path.join(BASEDIR,
                          'PLOS_paper',
                          'data',
                          'hypoxia_output.csv')

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
    "zero_flag": conf['zero_flag']
}

results = data_import(pfile)

true_medians= {'P_v': 4.0,
 'R_auto': 1.5,
 'Xtot': 9.1,
 'mu_max': 1.0,
 'n_h': 2.5,
 'n_m': 1.83,
 'phi': 0.036,
 'r_m': 0.027,
 'r_t': 0.018,
 'sigma_coll': 62.79}

# print(results.columns)


# Set accepted limit, lim
# lims = [1000]
tols = [0.11]
distances = []
for dist_measure in ['NRMSE']:
    # distances.extend(['{}_{}'.format(t, dist_measure)
    #                   for t in config['targets']])
    distances.append(dist_measure)
# for lim in lims:
for tol in tols:
    for d in distances:
        print("Working on {}".format(d.upper()))
        figPath = "/home/buck06191/Dropbox/phd/Bayesian_fitting/{}/{}/{}/{}/{}/{}/{}/"\
            "Figures/{}".format(model_name, 'PLOS_paper', 'hypoxia',
                                'healthy', 'wide_range', 'inflection', 'tolerance', d)

        dir_util.mkpath(figPath)
        print("Plotting total histogram")
        hist1 = histogram_plot(results, distance=d, frac=1)
        hist1.savefig(
            os.path.join(figPath, 'full_histogram_healthy.png'),
            bbox_inches='tight')
        print("Plotting fraction histogram")
        hist2 = histogram_plot(results, distance=d, tolerance=tol)
        hist2.savefig(
            os.path.join(
                figPath, 'tolerance_{}_histogram_healthy.png'.format(str(tol).replace('.', '_'))),
            bbox_inches='tight')
        print("Considering values below {}".format(tol))
        # print("Generating scatter plot")
        # scatter_dist_plot(results, params, limit=lim, n_ticks=4)
        print("Generating KDE plot")
        
        g = kde_plot(results, params, tolerance=tol, n_ticks=4, d=d,
                     median_file=os.path.join(figPath, "medians.txt"),
                     true_medians=true_medians)
        g.fig.savefig(
            os.path.join(figPath, 'PLOS_healthy_{}_{}_kde.png'
                         .format(str(tol).replace('.', '_'), d)),
            bbox_inches='tight')
        print("Generating averaged time series plot")
        config["offset"] = {}
        for t in config["targets"]:
            config["offset"]["{}_offset".format(t)] = d0[t][0]
        fig = plot_repeated_outputs(results, n_repeats=25, tolerance=tol,
                                    distance=d, **config)
        fig.set_size_inches(18.5, 12.5)
        fig.savefig(
            os.path.join(figPath, 'PLOS_healthy_{}_{}_TS.png'
                         .format(str(tol).replace('.', '_'), d)),
            dpi=100)
        plt.close('all')

# TODO: Fix issue with plot formatting, cutting off axes etc
# TODO: Fix issue with time series cutting short.
