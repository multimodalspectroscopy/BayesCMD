import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import os
from math import log10, floor
import random
import sys
import argparse
sys.path.append('../../..')
from bayescmd.util import findBaseDir
from bayescmd import abc
from bayescmd.bcmdModel import ModelBCMD
sns.set_context('talk')
sns.set_style('ticks')
BASEDIR = os.path.abspath(findBaseDir('BayesCMD'))
os.environ['BASEDIR']='BayesCMD'

def round_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def data_import(pfile, nan_sub=100000, chunk_size=10000, verbose=True):
    """
    Function to import
    :param pfile: Path to the file of parameters and distances
    :type pfile: str
    :param nan_sub: Number to substitute for NaN distances/params
    :type nan_sub: int or float
    :param chunk_size: Size of chunks to load for dataframe
    :type: int
    :param verbose: Boolean as to whether include verbose information.
    :type verbose: boolean
    :return: Dataframe containing all the parameters and distances, with NaN swapped for nan_sub
    :rtype: pd.DataFrame
    """

    result = pd.DataFrame()

    num_lines = sum(1 for line in open(pfile))-1
    df = pd.read_csv(pfile, chunksize=chunk_size, index_col='idx')
    for chunk in df:
        chunk.fillna(nan_sub, inplace=True)
        result = result.append(chunk)
    if verbose:
        print("Number of lines:\t{}".format(num_lines))
        print("Number of NaN values:\t{}".format(num_lines-sum(pd.notnull(result['euclidean']))))

    return result

def frac_calculator(df, frac):
    """

    :param df: Data frame to find fraction of
    :type df: pd.DataFrame
    :param frac: fraction of results to consider. Should be given as a percentage i.e. 1=1%, 0.1=0.1%
    :type frac: float
    :return: Number of values that make up the fraction
    :rtype: int
    """
    return int(len(df) * frac / 100)

def histogram_plot(df, distance='euclidean', fraction = 1, n_bins=100):
    """
    Function to plot histogram of distance values
    :param df: Dataframe of distances and parameters, generated using :func:`data_import`
    :type df: pd.DataFrame
    :param distance: Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'
    :type distance: str
    :param fraction: Fraction of all distances to plot. Varies from 0 to 1.
    :type fraction: float
    :param n_bins: Number of histogram bins
    :type n_bins: int
    :return: Plot Object
    """
    sorted_df = df.sort_values(by=distance)
    ax = plt.subplot(111, xlabel='Error - {}'.format(distance),
                     title='Distribution of {} Error Values'.format(distance.capitalize()))
    return sorted_df[distance].head(int(len(sorted_df)*fraction)).plot(kind='hist', bins=n_bins, ax=ax)


def scatter_dist_plot(df, params, frac, n_ticks, d=r'euclidean', verbose=False):
    """

    Inputs:
    =======
    :param df: Dataframe of distances and parameters, generated using :func:`data_import`
    :type df: pd.DataFrame
    :param params: list of params to compare pairwise
    :type params: list
    :param frac: fraction of results to consider. Should be given as a percentage i.e. 1=1%, 0.1=0.1%
    :type frac: float
    :param n_ticks: number of x-axis ticks
    :type n_ticks: int
    :param d: distance measure
    :type d: str
    :param verbose: Boolean to indicate verbosity
    :type verbose: boolean
    """

    sorted_df = df.sort_values(by=d)

    accepted_limit = frac_calculator(df, frac)
    plt.figure()

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
    sorted_df['Accepted'][sorted_df[d]==100000] = 2
    sorted_df['Accepted'] = sorted_df['Accepted'].astype('category')
    if verbose:
        print(sorted_df['Accepted'].value_counts())
    color_pal = {0: 'b', 1: 'g', 2: 'r'}
    with sns.plotting_context("talk", font_scale=1.2, rc={"figure.figsize":(12,9)}):
        g = sns.PairGrid(sorted_df, vars=params, hue='Accepted', size=4, diag_sharey=False)
        g.map_diag(sns.kdeplot, lw=3, legend=False)
        g.map_offdiag(plt.scatter, s=1)
        g.add_legend()
        for ii, ax in enumerate(g.axes.flat):
            xmax = max(sorted_df[params[ii%len(params)]]) #TODO: Need to ensure xmin/xmax equal prior limits
            xmin = min(sorted_df[params[ii%len(params)]])
            xticks = np.arange(xmin, xmax, round_sig((xmax-xmin)/n_ticks))
            ax.set_xticks(xticks)
            for label in ax.get_xticklabels():
                label.set_rotation(50)
        plt.subplots_adjust(top=0.9)
        plt.suptitle("Parameter distributions - Top {}% based on {} distance".format(frac, d))
        #new_labels = [r'Yes', r'No', r'Fail']
        #for t, l in zip(g.fig.get_children()[-1].texts, new_labels):
        #    t.set_text(l)
        lgd = g.fig.get_children()[-1]
        for i in range(2):
            lgd.legendHandles[i].set_sizes([50])
        
            
        return g

def kde_plot(df, params,frac, plot_param=1, n_ticks=6, d=r'euclidean', verbose=False):

    """

    Inputs:
    =======
    :param df: Dataframe of distances and parameters, generated using :func:`data_import`
    :type df: pd.DataFrame
    :param params: list of params to compare pairwise
    :type params: list
    :param frac: fraction of results to consider. Should be given as a percentage i.e. 1=1%, 0.1=0.1%
    :type frac: float
    :param plot_param: Which group to plot: 0: Outside posterior, 1: Inside posterior, 2: Failed run
    :param n_ticks: number of x-axis ticks
    :type n_ticks: int
    :param d: distance measure
    :type d: str
    :param verbose: Boolean as to whether include verbose information.
    :type verbose: boolean
    """

    sorted_df = df.sort_values(by=d)

    accepted_limit = frac_calculator(df, frac)

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
    sorted_df['Accepted'][sorted_df[d]==100000] = 2
    color_pal = {0: 'b', 1: 'g', 2: 'r'}
    kde_df=sorted_df.loc[(sorted_df['Accepted']==plot_param), :]
    if verbose:
        print(kde_df['Accepted'].value_counts())

    
    g=sns.PairGrid(kde_df, vars=params, hue='Accepted', palette=color_pal, diag_sharey=False)
    g.map_diag(sns.kdeplot)
    g.map_lower(sns.kdeplot, lw=3)
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    for ii, ax in enumerate(g.axes.flat):
        for label in ax.get_xticklabels():
            label.set_rotation(50)
        ax.set_ylim(0.8*min(sorted_df[params[ii//len(params)]]), 1.2*max(sorted_df[params[ii//len(params)]]))
        ax.set_xlim(0.8*min(sorted_df[params[ii%len(params)]]), 1.2*max(sorted_df[params[ii%len(params)]]))
    plt.subplots_adjust(top=0.9)
    title_dict = {0: "Outside Posterior", 1: "Inside Posterior", 2: "Failed Run"}
    plt.suptitle("Parameter distributions - Top {}% based on {} ({})".format(frac, d, title_dict[plot_param]))
    return g



def run_model(model):
    model.create_initialised_input()
    model.run_from_buffer()
    output = model.output_parse()
    return output

def get_output(model_name, p, input_data, d0, targets, distance='euclidean', zero_flag=None):
    """
    Function to generate model output and distances
    :param model_name: Name of model
    :type model_name: str
    :param p: Dict of {parameters: values} for which posteriors are being investigated
    :type p: dict
    :param input_data: Input data dict as generated by :method:`abc.inputParse`
    :type input_data: dict
    :param inputs: List of model inputs
    :type inputs: list
    :param targets: Model outputs against which the model is being optimised
    :type targets: list
    :param distance: Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'
    :type distance: str
    :param zero_flag: None/List of 0,1 identifying which targets need to be zero'd before calculating distance
    :type zero_flag: None or list
    :return: tuple of (p, model output data)
    """

    times = input_data['t']

    model = ModelBCMD(model_name,
                     inputs=input_data,
                     params=p,
                     times=times,
                     outputs=targets)

    output = run_model(model)

    dist = abc.get_distance(d0,
                            output,
                            targets,
                            distance=distance,
                            zero_flag=zero_flag)

    for k, v in dist.items():
        p[k] = v

    return p, output


def plot_repeated_outputs(df, model_name, parameters, input_path, inputs, openopt_path, targets, n_repeats, frac,
                          distance='euclidean', zero_flag=None):
    """
    Function to generate model output and distances
    :param model_name: Name of model
    :type model_name: str
    :param parameters: List of parameters for which posteriors are being investigated
    :type parameters: list
    :param input_path: Path to the true data file
    :type input_path: str
    :param inputs: List of model inputs
    :type inputs: list
    :param targets: Model outputs against which the model is being optimised
    :type targets: list
    :param n_repeats: Number of times to generate output data
    :type n_repeats: int
    :param frac: fraction of results to consider. Should be given as a percentage i.e. 1=1%, 0.1=0.1%
    :type frac: float
    :param distance: Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'
    :type distance: str
    :param zero_flag: None/List of 0,1 identifying which targets need to be zero'd before calculating distance
    :type zero_flag: None or list
    :return:
    """
    outputs_list = []
    accepted_limit = frac_calculator(df, frac)
    d0 = abc.import_actual_data(input_path)
    input_data = abc.inputParse(d0, inputs)

    openopt_data = pd.read_csv(openopt_path)

    try:
        rand_selection = random.sample(range(accepted_limit), n_repeats)
    except ValueError as e:
        print("Numer of requested model runs greater than posterior size:"
              "\n\tPosterior Size: {]\n\tNumber of runs: {}".format(accepted_limit, n_repeats),file=sys.stderr)

    posteriors = df.iloc[:accepted_limit][parameters].as_matrix()
    while len(outputs_list) < n_repeats:
        p = dict(zip(parameters, posteriors[rand_selection.pop()]))
        output = get_output(model_name, p, input_data, d0, targets, distance='euclidean', zero_flag=None)
        outputs_list.append(output)

    times = input_data['t']

    d = {"Errors":{}, "Outputs":{}}


    d['Errors']['Average'] = np.median([o[0]['TOTAL'] for o in outputs_list])
    for target in targets:
        d['Errors'][target] = np.median([o[0][target] for o in outputs_list])
        d['Outputs'][target] = [o[1][target] for o in outputs_list]

    with sns.plotting_context("talk", rc={"figure.figsize": (12, 9)}):
        fig, ax = plt.subplots(len(targets))
        for ii, target in enumerate(targets):
            g = sns.tsplot(data=d['Outputs'][target], time=times, estimator=np.median, ax=ax[ii])
            true_plot, = ax[ii].plot(times, input_data[target], 'g', label='True')
            openopt_plot, = ax[ii].plot(times, openopt_data[target], 'r', label='OpenOpt')
            bayes_line = mlines.Line2D([], [], color=sns.color_palette()[0], label='Bayes')
            ax[ii].set_title("{}: Average Euclidean Distance of {:.4f}".format(target, d['Errors'][target]))
            ax[ii].set_ylabel(r'{}'.format(target))
            ax[ii].set_xlabel('Time (sec)')
            ax[ii].title.set_fontsize(19)
            for item in ([ax[0].xaxis.label, ax[0].yaxis.label] +
                             ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                item.set_fontsize(17)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Simulated output for {} repeats using top {}% of data".format(n_repeats, frac))
        ax[0].legend(handles=[bayes_line, true_plot, openopt_plot], prop={"size": 17}, bbox_to_anchor=(1.25, 1.5))
    return None



if __name__=="__main__:":

    ap = argparse.ArgumentParser('Choose model to batch run:')
    ap.add_argument('input_file', metavar="INPUT_FILE", help='File containing parameters from multiple runs')

    args = ap.parse_args()

    pfile = args.input_file  # 'concatenated_results_190917.csv'
    params = [r'r_t', r'r_0', r'r_m', r'cytox_tot_tis', r'Vol_mit', r'O2_n', r'v_cn', r'sigma_coll']

    input_path = os.path.join(BASEDIR, 'data', 'hx01.csv')
    openopt_path = os.path.join(BASEDIR, 'data', 'model_run_output.csv')

    targets = ['Vmca', 'CCO']
    model_name = 'BS'
    inputs = ['Pa_CO2', 'P_a', 'SaO2sup']

    config = {"model_name": model_name,
              "targets": targets,
              "inputs": inputs,
              "parameters": params,
              "openopt_path": openopt_path,
              "input_path": input_path}

    results = data_import(pfile)

    for f in [0.01, 0.1, 1.0]:
        scatter_dist_plot(results, params, f, 6)
        plt.show()
        g = kde_plot(results, params, f)
        plt.show()
        plot_repeated_outputs(results, n_repeats=75, frac=f, **config)
        plt.show()

    rt_range = np.linspace(0.01, 0.03, 100)
    rm_range = np.linspace(0.01, 0.04, 100)
    r0_range = np.linspace(0.007, 0.0175, 100)
    volmit_range = np.linspace(0.02, 0.12, 100)
    cytox_range = np.linspace(0.0025, 0.009, 100)

    ranges = [rt_range, rm_range, r0_range, volmit_range, cytox_range]


