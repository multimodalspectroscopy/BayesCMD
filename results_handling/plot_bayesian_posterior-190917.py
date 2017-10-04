import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import scipy.integrate
import os
from math import log10, floor
import random
import sys
import argparse
sys.path.append('..')
os.environ['BASEDIR'] = 'BayesCMD'
from bayescmd.util import findBaseDir
sns.set_context('talk')
sns.set_style('ticks')
BASEDIR = os.path.abspath(findBaseDir('BayesCMD'))
from bayescmd import abc
from bayescmd.bcmdModel import ModelBCMD


def round_sig(x, sig=1):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


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

    num_lines = sum(1 for line in open(pfile)) - 1
    df = pd.read_csv(pfile, chunksize=chunk_size, index_col='idx')
    for chunk in df:
        chunk.fillna(nan_sub, inplace=True)
        result = result.append(chunk)
    if verbose:
        print("Number of lines:\t{}".format(num_lines))
        print("Number of NaN values:\t{}".format(
            num_lines - sum(pd.notnull(result['euclidean']))))

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


def histogram_plot(df, distance='euclidean', fraction=1, n_bins=100):
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
    ax = plt.subplot(
        111,
        xlabel='Error - {}'.format(distance),
        title='Distribution of'
        '{} Error Values'.format(distance.capitalize()))
    return sorted_df[distance].head(int(len(sorted_df) * fraction)).plot(
        kind='hist', bins=n_bins, ax=ax)


def scatter_dist_plot(df, params, frac, n_ticks, d=r'euclidean',
                      verbose=False):
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

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
    sorted_df['Accepted'][sorted_df[d] == 100000] = 2
    sorted_df['Accepted'] = sorted_df['Accepted'].astype('category')
    if verbose:
        print(sorted_df['Accepted'].value_counts())
    color_pal = {0: 'b', 1: 'g', 2: 'r'}
    with sns.plotting_context("talk", font_scale=1.2):
        g = sns.PairGrid(
            sorted_df, vars=params, hue='Accepted', diag_sharey=False)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(plt.scatter, s=1)
        g.add_legend()
        for ii, ax in enumerate(g.axes.flat):
            # TODO: Need to ensure xmin/xmax equal prior limits
            xmax = max(sorted_df[params[ii % len(params)]])
            xmin = min(sorted_df[params[ii % len(params)]])
            xticks = np.arange(xmin, xmax, round_sig((xmax - xmin) / n_ticks))
            ax.set_xticks(xticks)
            for label in ax.get_xticklabels():
                label.set_rotation(50)
        plt.subplots_adjust(top=0.9)
        plt.suptitle("Parameter distributions - Top {}% based on {} distance".
                     format(frac, d))
        #new_labels = [r'Yes', r'No', r'Fail']
        # for t, l in zip(g.fig.get_children()[-1].texts, new_labels):
        #    t.set_text(l)
        lgd = g.fig.get_children()[-1]
        for i in range(2):
            lgd.legendHandles[i].set_sizes([50])

        g.fig.tight_layout()
        return g


def diag_kde_plot(x, **kws):
    ax = plt.gca()
    p = sns.distplot(x, ax=ax, hist_kws={"linewidth": 1})
    x1, y1 = p.get_lines()[0].get_data()
    # care with the order, it is first y
    # initial fills a 0 so the result has same length than x
    cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
    nearest_05 = np.abs(cdf - 0.5).argmin()
    x_median = x1[nearest_05]

    ax.text(
        0.2,
        0.9,
        "Median: {}".format(str(round_sig(x_median, 2))),
        transform=ax.transAxes)
    ax.vlines(x_median, 0, ax.get_ylim()[1])

    return ax


def kde_plot(df,
             params,
             frac,
             plot_param=1,
             n_ticks=6,
             d=r'euclidean',
             verbose=False):
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
    sorted_df['Accepted'][sorted_df[d] == 100000] = 2
    color_pal = {0: 'b', 1: 'g', 2: 'r'}
    kde_df = sorted_df.loc[(sorted_df['Accepted'] == plot_param), :]
    if verbose:
        print(kde_df['Accepted'].value_counts())

    g = sns.PairGrid(
        kde_df,
        vars=params,
        hue='Accepted',
        palette=color_pal,
        diag_sharey=False)
    g.map_diag(diag_kde_plot)
    g.map_lower(sns.kdeplot, lw=3)
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    for ii, ax in enumerate(g.axes.flat):
        for label in ax.get_xticklabels():
            label.set_rotation(50)
        ax.set_ylim(
            min(sorted_df[params[ii // len(params)]]),
            max(sorted_df[params[ii // len(params)]]))
        ax.set_xlim(
            min(sorted_df[params[ii % len(params)]]),
            max(sorted_df[params[ii % len(params)]]))
    plt.subplots_adjust(top=0.9)
    title_dict = {
        0: "Outside Posterior",
        1: "Inside Posterior",
        2: "Failed Run"
    }
    plt.suptitle("Parameter distributions - Top {}% based on {} ({})".format(
        frac, d, title_dict[plot_param]))

    g.fig.tight_layout()
    return g


def run_model(model):
    model.create_initialised_input()
    model.run_from_buffer()
    output = model.output_parse()
    return output


def get_output(model_name,
               p,
               times,
               input_data,
               d0,
               targets,
               distance='euclidean',
               zero_flag=None):
    """
    Function to generate model output and distances
    :param model_name: Name of model
    :type model_name: str
    :param p: Dict of {parameters: values} for which posteriors are being investigated
    :type p: dict
    :param times: List of times at which the data was collected.
    :type times: list
    :param input_data: Input data dict as generated by :method:`abc.inputParse`
    :type input_data: dict
    :param targets: Model outputs against which the model is being optimised
    :type targets: list
    :param distance: Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'
    :type distance: str
    :param zero_flag: None/List of 0,1 identifying which targets need to be zero'd before calculating distance
    :type zero_flag: None or list
    :return: tuple of (p, model output data)
    """

    model = ModelBCMD(
        model_name, inputs=input_data, params=p, times=times, outputs=targets)

    output = run_model(model)

    dist = abc.get_distance(
        d0,
        output,
        targets,
        distance=distance,
        zero_flag=zero_flag,
        normalise=False)

    for k, v in dist.items():
        p[k] = v

    return p, output


def plot_repeated_outputs(df,
                          model_name,
                          parameters,
                          input_path,
                          inputs,
                          openopt_path,
                          targets,
                          n_repeats,
                          frac,
                          distance='euclidean',
                          zero_flag=None):
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

    sorted_df = df.sort_values(by=distance)

    outputs_list = []
    posterior_size = frac_calculator(df, frac)
    if n_repeats > posterior_size:
        print(
            "Setting number of repeats to quarter of the posterior size\n",
            file=sys.stderr)
        n_repeats = int(posterior_size / 4)
    d0 = abc.import_actual_data(input_path)
    input_data = abc.inputParse(d0, inputs)

    true_data = pd.read_csv(input_path)
    times = true_data['t'].as_matrix()

    openopt_data = pd.read_csv(openopt_path)

    try:
        rand_selection = random.sample(range(posterior_size), n_repeats)
    except ValueError as e:
        print(
            "Number of requested model runs greater than posterior size:"
            "\n\tPosterior Size: {}\n\tNumber of runs: {}".format(
                posterior_size, n_repeats),
            file=sys.stderr)
        raise e

    posteriors = sorted_df.iloc[:posterior_size][parameters].as_matrix()
    while len(outputs_list) < n_repeats:
        idx = rand_selection.pop()
        p = dict(zip(parameters, posteriors[idx]))
        output = get_output(
            model_name,
            p,
            times,
            input_data,
            d0,
            targets,
            distance='euclidean',
            zero_flag=None)
        outputs_list.append(output)

    d = {"Errors": {}, "Outputs": {}}

    d['Errors']['Average'] = np.nanmean([o[0]['TOTAL'] for o in outputs_list])
    for target in targets:
        d['Errors'][target] = np.nanmean([o[0][target] for o in outputs_list])
        print([any(np.isnan(o[1][target])) for o in outputs_list])
        for o in outputs_list:
            if any(np.isnan(o[1][target])):
                print(o[0])
        d['Outputs'][target] = [o[1][target] for o in outputs_list]

    with sns.plotting_context("talk", rc={"figure.figsize": (12, 9)}):
        fig, ax = plt.subplots(len(targets))
        for ii, target in enumerate(targets):
            g = sns.tsplot(
                data=d['Outputs'][target],
                time=times,
                estimator=np.median,
                ax=ax[ii])
            true_plot, = ax[ii].plot(
                times, true_data[target], 'g', label='True')
            openopt_plot, = ax[ii].plot(
                times, openopt_data[target], 'r', label='OpenOpt')
            bayes_line = mlines.Line2D(
                [], [], color=sns.color_palette()[0], label='Bayes')
            ax[ii].set_title("{}: Average Euclidean Distance of {:.4f}".format(
                target, d['Errors'][target]))
            ax[ii].set_ylabel(r'{}'.format(target))
            ax[ii].set_xlabel('Time (sec)')
            ax[ii].title.set_fontsize(19)
            for item in ([ax[0].xaxis.label, ax[0].yaxis.label] +
                         ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                item.set_fontsize(17)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Simulated output for {} repeats using top {}% of data".
                     format(n_repeats, frac))
        ax[0].legend(
            handles=[bayes_line, true_plot, openopt_plot],
            prop={"size": 17},
            bbox_to_anchor=(1.25, 1.5))
    return None


if __name__ == '__main__':

    ap = argparse.ArgumentParser('Choose model to batch run:')
    ap.add_argument(
        'input_file',
        metavar="INPUT_FILE",
        help='File containing parameters from multiple runs')

    args = ap.parse_args()

    pfile = args.input_file  # 'concatenated_results_190917.csv'
    params = [
        r'r_t', r'r_0', r'r_m', r'cytox_tot_tis', r'Vol_mit', r'O2_n', r'v_cn',
        r'sigma_coll'
    ]

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

    # histogram_plot(results)
    # plt.show()
    #histogram_plot(results, fraction=0.01)
    # plt.show()
    for f in [0.01, 0.1, 1.0]:
        print("Considering lowest {}% of values".format(f))
        #print("Generating scatter plot")
        #scatter_dist_plot(results, params, f, 6)
        # plt.show()
        #print("Generating KDE plot")
        #g = kde_plot(results, params, f)
        # plt.show()
        print("Generating averaged time series plot")
        plot_repeated_outputs(results, n_repeats=75, frac=f, **config)
        plt.show()

    # TODO: Fix issue with plot formatting, cutting off axes etc
    # TODO: Fix issue with time series cutting short.
    rt_range = np.linspace(0.01, 0.03, 100)
    rm_range = np.linspace(0.01, 0.04, 100)
    r0_range = np.linspace(0.007, 0.0175, 100)
    volmit_range = np.linspace(0.02, 0.12, 100)
    cytox_range = np.linspace(0.0025, 0.009, 100)

    ranges = [rt_range, rm_range, r0_range, volmit_range, cytox_range]
