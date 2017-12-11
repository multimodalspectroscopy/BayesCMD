"""Process results obtained using BayesCMD.

Process the various results obtained using BayesCMD, such as the
`parameters.csv` file. It is also possible to concatenate a number of different
`parameters.csv` files obtained using parallel batch runs into a single
parameters file.

Attributes
----------
BAYESCMD : :obj:`str`
    Absolute path to base directory. Found using
    :obj:`bayescmd.util.findBaseDir`

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import scipy.integrate
import os
import random
import sys
import pprint
sys.path.append('..')
from .util import findBaseDir  # noqa
from .util import round_sig  # noqa
sns.set_context('talk')
sns.set_style('ticks')
BASEDIR = os.path.abspath(findBaseDir('BayesCMD'))
from . import abc  # noqa
from .bcmdModel import ModelBCMD  # noqa


def data_merge(parent_directory, verbose=True):
    """Merge a set of parameters.csv files into one.

    Parameters
    ----------
    parent_directory : :obj:`list` of :obj:`str`
        Parent directory to a set of directories each containing model runs and
        a parameters.csv file.
    verbose : :obj:`boolean`, optional
        Boolean indicator of whether to print extra information.

    Returns
    -------
    None
        Concatenated will be written to file in `parent_directory`

    """
    dirs = os.listdir(parent_directory)
    if verbose:
        print(dirs)
    dfs = []
    for d in dirs:
        try:
            dfs.append(pd.read_csv(os.path.join(d, 'parameters.csv')))
        except FileNotFoundError:
            print("No parameters file in {}".format(d))
            continue

    for ii in range(len(dfs)):
        dfs[ii]['idx'] = dfs[ii].index.values + (len(dfs[ii]) * ii)
    df = pd.concat(dfs)
    df.index = range(len(df))
    df.to_csv(
        os.path.join(parent_directory, 'concatenated_results_150917.csv'),
        index=False)

    return None


def data_import(pfile, nan_sub=100000, chunk_size=10000, verbose=True):
    """Import a parameters file produced by a batch process.

    Parameters
    ----------
    pfile : str
        Path to the file of parameters and distances
    nan_sub: int or float, optional
        Number to substitute for NaN distances/params. Default of 100000
    chunk_size : int, optional
        Size of chunks to load for dataframe. Default of 10000
    verbose : bool, optional
        Boolean as to whether include verbose information. Default of True

    Returns
    -------
    result : pd.DataFrame
        Dataframe containing all the parameters and distances, with NaN swapped
        for nan_sub

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
    """Calculate the number of lines for a given fraction.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to find fraction of. Normally the output of
        :obj:`data_import`
    frac : float
        The fraction of results to consider. Should be given as a percentage
        i.e. 1=1%, 0.1=0.1%

    Returns
    -------
    int
        Number of lines that make up the fraction.

    """
    return int(len(df) * frac / 100)


def histogram_plot(df, distance='euclidean', fraction=1, n_bins=100):
    """Plot histogram of distance values.

    Plot a histogram showing the distribution of distance values for a given
    fraction of all distances in the dataframe. Distance values will have been
    calculated during the batch process.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe of distances and parameters, generated using
        :func:`data_import`
    distance : :obj:`str`, optional
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.
        Default is 'euclidean'.
    fraction : :obj:`float`, optional
        Fraction of all distances to plot. Varies from 0 to 1. Default is 1.
    n_bins : :obj:`int`, optional
        Number of histogram bins. Default is 100.

    Returns
    -------
    matplotlib.figure
        Matplotlib figure with histogram on.

    """
    sorted_df = df.sort_values(by=distance)
    fig = plt.figure()
    ax = plt.subplot(
        111,
        xlabel='Error - {}'.format(distance),
        title='Distribution of '
        '{} Error Values'.format(distance.capitalize()))
    sorted_df[distance].head(int(len(sorted_df) * fraction)).plot(
        kind='hist', bins=n_bins, ax=ax)
    return fig


def scatter_dist_plot(df,
                      params,
                      frac,
                      n_ticks=6,
                      d=r'euclidean',
                      verbose=False):
    """Plot distribution of parameters as a scatter PairPlot.

    Parameters
    ----------
    df: :obj:`pandas.DataFrame`
        Dataframe of distances and parameters, generated using
        :func:`data_import`
    params : :obj:`dict` of :obj:`str`: :obj:`tuple`
        Dict of model parameters to compare, with value tuple of the prior max
        and min.
    frac : :obj:`float`
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
    n_ticks : :obj:`int`, optional
        Number of x-axis ticks. Useful when a large number of parameters are
        being compared, as the axes can become crowded if the number of ticks
        is too high.
    d : :obj:`str`, optional
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.

            Note: Should be given  as a raw string if latex is used i.e.
            `r'MAE'`.
    verbose : :obj:`boolean`, optional
        Boolean to indicate verbosity. Default is False.

    Returns
    -------
    g : :obj:`seaborn.PairGrid`
        Seaborn pairgrid object is returned in case of further formatting.

    """
    p_names = list(params.keys)
    sorted_df = df.sort_values(by=d)

    accepted_limit = frac_calculator(df, frac)

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
    sorted_df['Accepted'][sorted_df[d] == 100000] = 2
    sorted_df['Accepted'] = sorted_df['Accepted'].astype('category')
    if verbose:
        print(sorted_df['Accepted'].value_counts())

    with sns.plotting_context("talk", font_scale=1.2):
        g = sns.PairGrid(
            sorted_df, vars=p_names, hue='Accepted', diag_sharey=False)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(plt.scatter, s=1)
        g.add_legend()
        for ii, ax in enumerate(g.axes.flat):
            ii_y = ii // len(p_names)
            ii_x = ii % len(p_names)
            ax.set_ylim(params[p_names[ii_y]])
            ax.set_xlim(params[p_names[ii_x]])
            xmax = params[p_names[ii_x]][1]
            xmin = params[p_names[ii_x]][0]
            xticks = np.arange(xmin, xmax, round_sig((xmax - xmin) / n_ticks))
            ax.set_xticks(xticks)
            for label in ax.get_xticklabels():
                label.set_rotation(50)
        plt.subplots_adjust(top=0.9)
        plt.suptitle("Parameter distributions - Top {}% based on {} distance".
                     format(frac, d))
        # new_labels = [r'Yes', r'No', r'Fail']
        # for t, l in zip(g.fig.get_children()[-1].texts, new_labels):
        #    t.set_text(l)
        lgd = g.fig.get_children()[-1]
        for i in range(2):
            lgd.legendHandles[i].set_sizes([50])

        g.fig.tight_layout()
    return g


def diag_kde_plot(x, medians, true_medians, **kws):
    """Plot univariate KDE and barplot with median of distribution marked on.

    Includes median of distribution as a line and as text.

    Parameters
    ----------
    x : array-like
        Array-like of data to plot.
    medians : :obj:`dict`
        Dictionary of parameter, median pairings.
    kws : key, value pairings.
        Other keyword arguments to pass to :obj:`sns.distplot`.

    Returns
    -------
    ax : :obj:`matplotlib.AxesSubplot`
        AxesSubplot object of univariate KDE and bar plot with median marked
        on as well as text.

    """
    ax = plt.gca()
    p = sns.distplot(x, ax=ax, hist_kws={"linewidth": 1})
    x1, y1 = p.get_lines()[0].get_data()
    # care with the order, it is first y
    # initial fills a 0 so the result has same length than x
    cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
    nearest_05 = np.abs(cdf - 0.5).argmin()
    x_median = x1[nearest_05]
    medians[x.name] = round_sig(x_median, 2)
    ax.vlines(x_median, 0, ax.get_ylim()[1])
    if true_medians is not None:
        ax.vlines(true_medians[x.name], 0, ax.get_ylim()[1], 'r')
    ax.text(
        0.05,
        1.1,
        "Median: {:.2E}".format(x_median),
        verticalalignment='center',
        transform=ax.transAxes)

    return ax


def kde_plot(df,
             params,
             frac,
             true_medians=None,
             plot_param=1,
             n_ticks=6,
             d=r'euclidean',
             verbose=False):
    """Plot the model parameters pairwise as a KDE.

    Parameters
    ----------
    df: :obj:`pandas.DataFrame`
        Dataframe of distances and parameters, generated using
        :func:`data_import`
    params : :obj:`dict` of :obj:`str`: :obj:`tuple`
        Dict of model parameters to compare, with value tuple of the prior max
        and min.
    frac : :obj:`float`
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
    true_median : :obj:`dict` or :obj: `None`
        Dictionary of true median values if known.
    plot_param : :obj:`int`
        Which group to plot:

            0: Outside posterior
            1: Inside posterior
            2: Failed run
    n_ticks : :obj:`int`, optional
        Number of x-axis ticks. Useful when a large number of parameters are
        bring compared, as the axes can become crowded if the number of ticks
        is too high.
    d : :obj:`str`, optional
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.

            Note: Should be given  as a raw string if latex is used i.e.
            `r'MAE'`.
    verbose : :obj:`boolean`, optional
        Boolean to indicate verbosity. Default is False.

    Returns
    -------
    g : :obj:`seaborn.PairGrid`
        Seaborn pairgrid object is returned in case of further formatting.

    """
    p_names = list(params.keys())
    sorted_df = df.sort_values(by=d)

    accepted_limit = frac_calculator(df, frac)

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
    sorted_df['Accepted'][sorted_df[d] == 100000] = 2
    color_pal = {0: 'b', 1: 'g', 2: 'r'}
    kde_df = sorted_df.loc[(sorted_df['Accepted'] == plot_param), :]
    if verbose:
        print(kde_df['Accepted'].value_counts())
    with sns.plotting_context("talk", rc={"figure.figsize": (12, 9)}):
        g = sns.PairGrid(
            kde_df,
            vars=p_names,
            hue='Accepted',
            palette=color_pal,
            diag_sharey=False)
        medians = {}
        g.map_diag(diag_kde_plot, medians=medians, true_medians=true_medians)
        for k, v in medians.items():
            print("{}: {}".format(k, v))
        g.map_lower(sns.kdeplot, lw=3)
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            g.axes[i, j].set_visible(False)

        for ii, ax in enumerate(g.axes.flat):
            for label in ax.get_xticklabels():
                label.set_rotation(50)
            ii_y = ii // len(p_names)
            ii_x = ii % len(p_names)
            ax.set_ylim(params[p_names[ii_y]])
            ax.set_xlim(params[p_names[ii_x]])
            xmax = params[p_names[ii_x]][1]
            xmin = params[p_names[ii_x]][0]
            xticks = np.arange(xmin, xmax,
                               round_sig((xmax - xmin) / n_ticks, sig=1))
            ax.set_xticks(xticks)
        # plt.subplots_adjust(top=0.8)
        title_dict = {0: "(Outside Posterior)", 1: "", 2: "(Failed Run)"}
        plt.suptitle("Parameter distributions - Top {} %"
                     "based on {} {}".format(frac, d, title_dict[plot_param]))

        g.fig.tight_layout()
        g.fig.subplots_adjust(bottom=0.15, top=0.9)
    return g


def run_model(model):
    """Run a BCMD Model.

    Parameters
    ----------
    model : :obj:`bayescmd.bcmdModel.ModelBCMD`
        An initialised instance of a ModelBCMD class.

    Returns
    -------
    output : :obj:`dict`
        Dictionary of parsed model output.

    """
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
    """Generate model output and distances.

    Parameters
    ----------
    model_name : :obj:`str`
        Name of model
    p : :obj:`dict`
        Dict of form {'parameter': value} for which posteriors are being
        investigated.
    times : :obj:`list` of :obj:`float`
        List of times at which the data was collected.
    input_data : :obj:`dict`
        Dictionary of input data as generated by :obj:`abc.inputParse`.
    d0 : :obj:`dict`
        Dictionary of real data, as generated by :obj:`abc.import_actual_data`.
    targets : :obj:`list` of :obj:`str`
        List of model outputs against which the model is being optimised.
    distance : :obj:`str`
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.
    zero_flag : dict
        Dictionary of form target(:obj:`str`): bool, where bool indicates
        whether to zero that target.

        Note: zero_flag keys should match targets list.

    Returns
    -------
    :obj:`tuple`
        A tuple of (p, model output data).

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
                          targets,
                          n_repeats,
                          frac,
                          zero_flag,
                          openopt_path=None,
                          distance='euclidean'):
    """Generate model output and distances multiple times.

    Parameters
    ----------
    model_name : :obj:`str`
        Name of model. Should match the modeldef file for model being generated
        i.e. model_name of 'model`' should have a modeldef file
        'model1.modeldef'.
    parameters : :obj:`dict` of :obj:`str`: :obj:`tuple`
        Dict of model parameters to compare, with value tuple of the prior max
        and min.
    input_path : :obj:`str`
        Path to the true data file
    inputs : :obj:`list` of :obj:`str`
        List of model inputs.
    targets : :obj:`list` of :obj:`str`
        List of model outputs against which the model is being optimised.
    n_repeats : :obj: `int`
        Number of times to generate output data
    frac : :obj:`float`
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
    zero_flag : dict
        Dictionary of form target(:obj:`str`): bool, where bool indicates
        whether to zero that target.

        Note: zero_flag keys should match targets list.
    openopt_path : :obj:`str` or :obj:`None`
        Path to the openopt data file if it exists. Default is None.
    distance : :obj:`str`, optional
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.

    Returns
    -------
    fig : :obj:`matplotlib.figure`
        Figure containing all axes.

    """
    p_names = list(parameters.keys())
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

    if openopt_path:
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

    posteriors = sorted_df.iloc[:posterior_size][p_names].as_matrix()
    while len(outputs_list) < n_repeats:
        idx = rand_selection.pop()
        p = dict(zip(p_names, posteriors[idx]))
        output = get_output(
            model_name,
            p,
            times,
            input_data,
            d0,
            targets,
            distance='euclidean',
            zero_flag=zero_flag)
        outputs_list.append(output)

    d = {"Errors": {}, "Outputs": {}}

    d['Errors']['Average'] = np.nanmean([o[0]['TOTAL'] for o in outputs_list])
    for target in targets:
        d['Errors'][target] = np.nanmean([o[0][target] for o in outputs_list])
        d['Outputs'][target] = [o[1][target] for o in outputs_list]

    with sns.plotting_context(
            "talk", font_scale=1.6, rc={"figure.figsize": (24, 18)}):
        fig, ax = plt.subplots(len(targets))
        if type(ax) != np.ndarray:
            ax = np.asarray([ax])
        for ii, target in enumerate(targets):
            sns.tsplot(
                data=d['Outputs'][target],
                time=times,
                estimator=np.median,
                ci=95,
                ax=ax[ii])
            paths = []
            true_plot, = ax[ii].plot(
                times, true_data[target], 'g', label='True')
            paths.append(true_plot)
            if openopt_path:
                openopt_plot, = ax[ii].plot(
                    times, openopt_data[target], 'r', label='OpenOpt')
                paths.append(openopt_plot)
            bayes_line = mlines.Line2D(
                [], [], color=sns.color_palette()[0], label='Bayes')
            paths.append(bayes_line)
            ax[ii].set_title("{}: Average Euclidean Distance of {:.4f}".format(
                target, d['Errors'][target]))
            ax[ii].set_ylabel(r'{}'.format(target))
            ax[ii].set_xlabel('Time (sec)')
            ax[ii].title.set_fontsize(25)
            for item in ([ax[0].xaxis.label, ax[0].yaxis.label] +
                         ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                item.set_fontsize(22)
        ax[0].legend(
            handles=paths, prop={"size": 22}, bbox_to_anchor=(1.15, -0.5))
        plt.tight_layout()
        fig.suptitle("Simulated output for {} repeats using top {}% of data".
                     format(n_repeats, frac))
    return fig
