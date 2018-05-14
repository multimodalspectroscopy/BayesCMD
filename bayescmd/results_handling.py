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
import re
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


def sort_human(l):
    """Sort a list of strings by numerical."""
    def convert(text): return float(text) if text.isdigit() else text

    def alphanum(key): return [convert(c)
                               for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l


def data_merge_by_date(date, parent_directory, verbose=True):
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
    dirs = [os.path.abspath(os.path.join(parent_directory, f))
            for f in os.listdir(parent_directory) if (date in f) and
            (os.path.splitext(f)[1] != ".csv")]
    if verbose:
        print(dirs)
    dfs = []
    for ii, d in enumerate(dirs):
        try:
            dfs.append(pd.read_csv(os.path.join(d, 'parameters.csv')))
            print("Batch Number: {}".format(ii))
            if ii is not 0:
                dfs[ii]['ix'] = dfs[ii].index.values + \
                    dfs[ii - 1]['ix'].values[-1]
            else:
                dfs[ii]['ix'] = dfs[ii].index.values
            if os.path.split(d)[1][:4].isdigit():
                print(os.path.split(d)[1][:4])
                dfs[ii]['Start Date'] = os.path.split(d)[1][:4]
            else:
                continue
        except FileNotFoundError:
            print("No parameters file in {}".format(d))
            continue
    if verbose:
        print("{} dataframes  to be joined".format(len(dfs)))
    # for ii in range(len(dfs)):
        # if ii is not 0:
        #     dfs[ii]['ix'] = dfs[ii].index.values + dfs[ii - 1]['ix'].values[-1]
        # else:
        #     dfs[ii]['ix'] = dfs[ii].index.values
        # if os.path.split(dirs[ii])[1][:4].isdigit():
        #     print(os.path.split(dirs[ii])[1][:4])
        #     dfs[ii]['Start Time'] = os.path.split(dirs[ii])[1][:4]
        # else:
        #     continue
    df = pd.concat(dfs)
    df.index = range(len(df))
    output_file = os.path.join(parent_directory,
                               'concatenated_results_{}.csv'.format(date))
    df.to_csv(output_file, index=False)

    return output_file


def data_merge_by_batch(parent_directory, verbose=True):
    """Merge a set of parameters.csv files into one.

    This is intended for use with batch processes from Legion, with each batch
    being 1000 runs longand numbered with integer values.

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
    dirs = [os.path.abspath(os.path.join(parent_directory, d))
            for d in os.listdir(parent_directory)
            if os.path.isdir(os.path.abspath(
                os.path.join(parent_directory, d))) and d != 'archives']
    dirs = sort_human(dirs)
    if verbose:
        print(dirs)
    dfs = []
    for d in dirs:
        try:
            dfs.append(pd.read_csv(os.path.join(d, 'parameters.csv')))
            ii = len(dfs)-1
            print("Processing parameter file {}".format(ii))
            if ii is not 0:
                dfs[ii]['ix'] = dfs[ii].index.values + \
                    dfs[ii - 1]['ix'].values[-1]
            else:
                dfs[ii]['ix'] = dfs[ii].index.values

            if os.path.split(d)[1].split('_')[-1].isdigit():
                print(os.path.split(d)[1].split('_')[-1])
                dfs[ii]['Batch'] = int(os.path.split(d)[1].split('_')[-1])
            else:
                print("Batch number not found for {}".format(d))
                continue
        except FileNotFoundError:
            print("No parameters file in {}".format(d))
            continue
    if verbose:
        print("{} dataframes  to be joined".format(len(dfs)))
    # for ii in range(len(dfs)):
        # if ii is not 0:
        #     dfs[ii]['ix'] = dfs[ii].index.values + dfs[ii - 1]['ix'].values[-1]
        # else:
        #     dfs[ii]['ix'] = dfs[ii].index.values
        # if os.path.split(dirs[ii])[1][:4].isdigit():
        #     print(os.path.split(dirs[ii])[1][:4])
        #     dfs[ii]['Start Time'] = os.path.split(dirs[ii])[1][:4]
        # else:
        #     continue
    df = pd.concat(dfs)
    df.index = range(len(df))
    output_file = os.path.join(parent_directory,
                               'all_parameters.csv')
    df.to_csv(output_file, index=False)

    return output_file


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


def histogram_plot(df, distance='euclidean', frac=None, limit=None, n_bins=100):
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
    limit : :obj:int`, optional
        Integer value for the top N values to accept in rejection.
    frac: :obj:`float`, optional
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
        If `limit` is given it takes precedence.
    n_bins : :obj:`int`, optional
        Number of histogram bins. Default is 100.

    Returns
    -------
    matplotlib.figure
        Matplotlib figure with histogram on.

    """
    sorted_df = df.sort_values(by=distance)

    if limit:
        accepted_limit = limit
    elif frac:
        accepted_limit = frac_calculator(sorted_df, frac)
    else:
        raise ValueError('No limit or fraction given.')

    fig = plt.figure()
    ax = plt.subplot(111,
                     xlabel='Error - {}'.format(distance),
                     title='Distribution of '
                     '{} Error Values'.format(distance.capitalize()))
    sorted_df[distance].head(accepted_limit).plot(
        kind='hist', bins=n_bins, ax=ax)
    return fig


def scatter_dist_plot(df,
                      params,
                      limit=None,
                      frac=None,
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
    limit : :obj:int`, optional
        Integer value for the top N values to accept in rejection.
    frac: :obj:`float`, optional
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
        If `limit` is given it takes precedence.
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
    p_names = list(params.keys())
    sorted_df = df.sort_values(by=d)

    if limit:
        accepted_limit = limit
    elif frac:
        accepted_limit = frac_calculator(sorted_df, frac)
    else:
        raise ValueError('No limit or fraction given.')

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
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
            ax.set_ylim(params[p_names[ii_y]][1])
            ax.set_xlim(params[p_names[ii_x]][1])
            xmax = params[p_names[ii_x]][1][1]
            xmin = params[p_names[ii_x]][1][0]
            xticks = np.arange(xmin, xmax, round_sig((xmax - xmin) / n_ticks))
            ax.set_xticks(xticks)
            for label in ax.get_xticklabels():
                label.set_rotation(50)
        plt.subplots_adjust(top=0.9)
        plt.suptitle("Parameter distributions - Top {}% based on {} distance".
                     format(frac, d))
        new_labels = [r'Yes', r'No']
        for t, l in zip(g.fig.get_children()[-1].texts, new_labels):
            t.set_text(l)
        lgd = g.fig.get_children()[-1]
        for i in range(2):
            lgd.legendHandles[i].set_sizes([50])

        g.fig.tight_layout()
    return g


def medians_kde_plot(x, y, medians, true_medians, openopt_medians, **kws):
    """Plot bivariate KDE with median of distribution marked on.


    Parameters
    ----------
    x : array-like
        Array-like of data to plot.
    y : array-like
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
    ax = sns.kdeplot(x, y, ax=ax)
    # x1, y1 = p.get_lines()[0].get_data()
    # # care with the order, it is first y
    # # initial fills a 0 so the result has same length than x
    # cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
    # nearest_05 = np.abs(cdf - 0.5).argmin()
    # x_median = x1[nearest_05]
    # medians[x.name] = round_sig(x_median, 2)
    # ax.vlines(x_median, 0, ax.get_ylim()[1])

    x_median = x.median()
    y_median = y.median()
    ax.plot(x_median, y_median, 'kX')
    if true_medians is not None:
        ax.plot(true_medians[x.name], true_medians[y.name], 'gX')

    if openopt_medians is not None:
        ax.plot(openopt_medians[x.name], openopt_medians[y.name], 'rX')

    return ax


def diag_kde_plot(x, medians, true_medians, openopt_medians, **kws):
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
    # cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
    # nearest_05 = np.abs(cdf - 0.5).argmin()
    # x_median = x1[nearest_05]
    # medians[x.name] = round_sig(x_median, 2)
    x_median = np.median(x)
    ax.vlines(x_median, 0, ax.get_ylim()[1])
    if true_medians is not None:
        ax.vlines(true_medians[x.name], 0, ax.get_ylim()[1], 'g')

    if openopt_medians is not None:
        ax.vlines(openopt_medians[x.name], 0, ax.get_ylim()[1], 'r')

    ax.text(
        0.05,
        1.1,
        "Median: {:.2E}".format(x_median),
        verticalalignment='center',
        transform=ax.transAxes)

    return ax


def kde_plot(df,
             params,
             limit=None,
             frac=None,
             median_file=None,
             true_medians=None,
             openopt_medians=None,
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
    limit : :obj:int`, optional
        Integer value for the top N values to accept in rejection.
    frac: :obj:`float`, optional
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
        If `limit` is given it takes precedence.
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

    if limit:
        accepted_limit = limit
    elif frac:
        accepted_limit = frac_calculator(sorted_df, frac)
    else:
        raise ValueError('No limit or fraction given.')

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
    sorted_df['Accepted'][sorted_df[d] == 100000] = 2
    color_pal = {0: 'b', 1: 'g', 2: 'r'}
    kde_df = sorted_df.loc[(sorted_df['Accepted'] == plot_param), :]
    if verbose:
        print(kde_df['Accepted'].value_counts())
    with sns.plotting_context("talk",
                              rc={"figure.figsize": (12, 9)}):
        g = sns.PairGrid(
            kde_df,
            vars=p_names,
            hue='Accepted',
            palette=color_pal,
            diag_sharey=False)
        medians = {}
        g.map_diag(diag_kde_plot, medians=medians, true_medians=true_medians,
                   openopt_medians=openopt_medians)
        for k, v in medians.items():
            if median_file:
                with open(median_file, 'a') as mf:
                    print("{}: {}".format(k, v), file=mf)
            else:
                print("{}: {}".format(k, v))
        # g.map_lower(sns.kdeplot, lw=3)
        g.map_lower(medians_kde_plot, medians=medians,
                    true_medians=true_medians, openopt_medians=openopt_medians)
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            g.axes[i, j].set_visible(False)

        for ii, ax in enumerate(g.axes.flat):
            for label in ax.get_xticklabels():
                label.set_rotation(50)
            ii_y = ii // len(p_names)
            ii_x = ii % len(p_names)
            ax.set_ylim(params[p_names[ii_y]][1])
            ax.set_xlim(params[p_names[ii_x]][1])
            xmax = params[p_names[ii_x]][1][1]
            xmin = params[p_names[ii_x]][1][0]
            xticks = np.arange(xmin, xmax,
                               round_sig((xmax - xmin) / n_ticks, sig=1))
            ax.set_xticks(xticks)
        # plt.subplots_adjust(top=0.8)
        title_dict = {0: "(Outside Posterior)", 1: "", 2: "(Failed Run)"}
        if limit:
            plt.suptitle("Parameter distributions - Top {} points "
                         "based on {} {}".format(
                             limit, d, title_dict[plot_param]),
                         fontsize=32)
        elif frac:
            plt.suptitle("Parameter distributions - Top {}% "
                         "based on {} {}".format(
                             frac, d, title_dict[plot_param]),
                         fontsize=32)

        lines = []
        if true_medians:
            lines.append(('True', mlines.Line2D([], [], color='green')))
        if openopt_medians:
            lines.append(('OpenOpt', mlines.Line2D([], [], color='red')))
        lines.append(('Inferred', mlines.Line2D([], [], color='black')))

        g.fig.legend(labels=[l[0] for l in lines],
                     handles=[l[1] for l in lines],
                     bbox_to_anchor=(0.7, 0.7), loc=2, prop={"size": 32})

        g.fig.tight_layout()
        g.fig.subplots_adjust(bottom=0.15, top=0.9)
    return g

def single_kde_plot(df,
             params,
             limit=None,
             frac=None,
             median_file=None,
             true_medians=None,
             openopt_medians=None,
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
        and min - should only be a single parameter.
    limit : :obj:int`, optional
        Integer value for the top N values to accept in rejection.
    frac: :obj:`float`, optional
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
        If `limit` is given it takes precedence.
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

    if len(p_names)!=1:
        raise ValueError("Number of parameters is {}. Should be 1.".format(len(p_names)))
    sorted_df = df.sort_values(by=d)

    if limit:
        accepted_limit = limit
    elif frac:
        accepted_limit = frac_calculator(sorted_df, frac)
    else:
        raise ValueError('No limit or fraction given.')

    sorted_df['Accepted'] = np.zeros(len(sorted_df))
    sorted_df['Accepted'].iloc[:accepted_limit] = 1
    sorted_df['Accepted'][sorted_df[d] == 100000] = 2
    color_pal = {0: 'b', 1: 'g', 2: 'r'}
    kde_df = sorted_df.loc[(sorted_df['Accepted'] == plot_param), :]
    if verbose:
        print(kde_df['Accepted'].value_counts())
    with sns.plotting_context("talk",
                              rc={"figure.figsize": (12, 9)}):

        fig, ax = plt.subplots(1)
        xx = kde_df[p_names[0]].values
        ax = sns.kdeplot(xx, shade=True, ax=ax)
        medians = {p_names[0]: np.median(xx)}

        ax.vlines(np.median(xx), 0, ax.get_ylim()[1])

        if true_medians is not None:
            ax.vlines(true_medians[p_names[0]], 0, ax.get_ylim()[1], 'g')

        if openopt_medians is not None:
            ax.vlines(openopt_medians[p_names[0]], 0, ax.get_ylim()[1], 'r')

        ax.text(
            0.05,
            0.8,
            "Median: {:.2E}".format(np.median(xx)),
            verticalalignment='center',
            transform=ax.transAxes)

        for k, v in medians.items():
            if median_file:
                with open(median_file, 'a') as mf:
                    print("{}: {}".format(k, v), file=mf)
            else:
                print("{}: {}".format(k, v))

        for label in ax.get_xticklabels():
            label.set_rotation(50)

        #ax.set_ylim(params[p_names[0]][1])
        ax.set_xlim(params[p_names[0]][1])
        xmax = params[p_names[0]][1][1]
        xmin = params[p_names[0]][1][0]
        xticks = np.arange(xmin, xmax,
                           round_sig((xmax - xmin) / n_ticks, sig=1))
        ax.set_xticks(xticks)
        title_dict = {0: "(Outside Posterior)", 1: "", 2: "(Failed Run)"}
        if limit:
            ax.set_title("Posterior distribution for {} -\n Top {} points "
                         "based on {} {}".format(
                             p_names[0], limit, d, title_dict[plot_param]),
                         fontsize=20)
        elif frac:
            ax.set_title("Posterior distribution for {} -\n Top {}% "
                         "based on {} {}".format(
                             p_names[0], frac, d, title_dict[plot_param]),
                         fontsize=20)

        lines = []
        if true_medians:
            lines.append(('True', mlines.Line2D([], [], color='green')))
        if openopt_medians:
            lines.append(('OpenOpt', mlines.Line2D([], [], color='red')))
        lines.append(('Inferred', mlines.Line2D([], [], color='black')))

        fig.legend(labels=[l[0] for l in lines],
                     handles=[l[1] for l in lines],
                     bbox_to_anchor=(0.87, 0.7), loc=2, prop={"size": 14})

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15, top=0.8, right=0.9)
    return fig


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
        distance=distance.split("_")[-1],
        zero_flag=zero_flag)

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
                          zero_flag,
                          limit=None,
                          frac=None,
                          openopt_path=None,
                          offset=None,
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
    offset : :obj:`dict`
        Dictionary of offset parameters if they are needed
    distance : :obj:`str`, optional
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.

    Returns
    -------
    fig : :obj:`matplotlib.figure`
        Figure containing all axes.

    """
    p_names = list(parameters.keys())
    sorted_df = df.sort_values(by=distance)

    if limit:
        accepted_limit = limit
    elif frac:
        accepted_limit = frac_calculator(sorted_df, frac)
    else:
        raise ValueError('No limit or fraction given.')

    outputs_list = []
    if n_repeats > accepted_limit:
        print(
            "Setting number of repeats to quarter of the posterior size\n",
            file=sys.stderr)
        n_repeats = int(accepted_limit / 4)
    d0 = abc.import_actual_data(input_path)
    input_data = abc.inputParse(d0, inputs)

    true_data = pd.read_csv(input_path)
    times = true_data['t'].as_matrix()

    if openopt_path:
        openopt_data = pd.read_csv(openopt_path)

    try:
        rand_selection = random.sample(range(accepted_limit), n_repeats)
    except ValueError as e:
        print(
            "Number of requested model runs greater than posterior size:"
            "\n\tPosterior Size: {}\n\tNumber of runs: {}".format(
                accepted_limit, n_repeats),
            file=sys.stderr)
        raise e

    posteriors = sorted_df.iloc[:accepted_limit][p_names].as_matrix()
    while len(outputs_list) < n_repeats:
        idx = rand_selection.pop()
        p = dict(zip(p_names, posteriors[idx]))
        if offset:
            p = {**p, **offset}
        output = get_output(
            model_name,
            p,
            times,
            input_data,
            d0,
            targets,
            distance=distance,
            zero_flag=zero_flag)
        outputs_list.append(output)

    d = {"Errors": {}, "Outputs": {}}

    d['Errors']['Average'] = np.nanmean([o[0]['TOTAL'] for o in outputs_list])
    for target in targets:
        d['Errors'][target] = np.nanmean([o[0][target] for o in outputs_list])
        d['Outputs'][target] = [o[1][target] for o in outputs_list]

    with sns.plotting_context(
            "talk", font_scale=1.6, rc={"figure.figsize": (26, 12)}):
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
                times, true_data[target], 'r', label='True')
            paths.append(true_plot)
            if openopt_path:
                openopt_plot, = ax[ii].plot(
                    times, openopt_data[target], 'g', label='OpenOpt')
                paths.append(openopt_plot)
            bayes_line = mlines.Line2D(
                [], [], color=sns.color_palette()[0], label='Bayes')
            paths.append(bayes_line)
            ax[ii].set_title("{}: Average {} Distance of {:.4f}".format(
                target, distance, d['Errors'][target]))
            ax[ii].set_ylabel(r'{}'.format(target))
            ax[ii].set_xlabel('Time (sec)')
            ax[ii].title.set_fontsize(25)
            for item in ([ax[0].xaxis.label, ax[0].yaxis.label] +
                         ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                item.set_fontsize(22)
        if openopt_path:
            fig.legend(labels=['True\nData', 'OpenOpt', 'Posterior\nPredictive'],
                       handles=paths, prop={"size": 20},
                       bbox_to_anchor=(1.0, 0.6))
        else:
            fig.legend(labels=['True\nData', 'Posterior\nPredictive'],
                       handles=paths, prop={"size": 20},
                       bbox_to_anchor=(1.0, 0.6))
        plt.subplots_adjust(hspace=1, right=0.8)
        if limit:
            fig.suptitle("Simulated output for {} repeats using\ntop {} parameter combinations\n".
                         format(n_repeats, limit))
        elif frac:
            fig.suptitle("Simulated output for {} repeats using top {}% of data\n".
                         format(n_repeats, frac))
    return fig
