"""Process results obtained using BayesCMD.

Process the various results obtained using BayesCMD, such as the
`parameters.csv` file. It is also possible to concatenate a number of different
`parameters.csv` files obtained using parallel batch runs into a single
parameters file.

Attributes
----------
BAYESCMD : :obj:`str`
    Absolute path to base directory. Found using
    :method:`bayescmd.util.findBaseDir()`

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
        :method:`data_import`
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
    matplotlib.AxesSubplot
        AxesSubplotobject that contains histogram of distance values.

    """
    sorted_df = df.sort_values(by=distance)
    ax = plt.subplot(
        111,
        xlabel='Error - {}'.format(distance),
        title='Distribution of '
        '{} Error Values'.format(distance.capitalize()))
    return sorted_df[distance].head(int(len(sorted_df) * fraction)).plot(
        kind='hist', bins=n_bins, ax=ax)


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
    params : :obj:`list` of :obj:`str`
        List of model parameters to compare pairwise.
    frac : :obj:`float`
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
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
        # new_labels = [r'Yes', r'No', r'Fail']
        # for t, l in zip(g.fig.get_children()[-1].texts, new_labels):
        #    t.set_text(l)
        lgd = g.fig.get_children()[-1]
        for i in range(2):
            lgd.legendHandles[i].set_sizes([50])

        g.fig.tight_layout()
    return g


def diag_kde_plot(x, **kws):
    """Plot univariate KDE and barplot with median of distribution marked on.

    Includes median of distribution as a line and as text.

    Parameters
    ----------
    x : array-like
        Array-like of data to plot.
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
    """Plot the model parameters pairwide as a KDE.

    Parameters
    ----------
    df: :obj:`pandas.DataFrame`
        Dataframe of distances and parameters, generated using
        :func:`data_import`
    params : :obj:`list` of :obj:`str`
        List of model parameters to compare pairwise.
    frac : :obj:`float`
        Fraction of results to consider. Should be given as a percentage i.e.
        1=1%, 0.1=0.1%
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
        Dictionary of input data as generated by :method:`abc.inputParse`.
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
                          openopt_path,
                          targets,
                          n_repeats,
                          frac,
                          distance='euclidean',
                          zero_flag=None):
    """Generate model output and distances multiple times.

    Parameters
    ----------
    model_name : :obj:`str`
        Name of model. Should match the modeldef file for model being generated
        i.e. model_name of 'model`' should have a modeldef file
        'model1.modeldef'.
    parameters : :obj:`list` of :obj:`str`
        List of parameters for which posteriors are being investigated.
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
    distance : :obj:`str`
        Distance measure. One of 'euclidean', 'manhattan', 'MAE', 'MSE'.
    zero_flag : dict
        Dictionary of form target(:obj:`str`): bool, where bool indicates
        whether to zero that target.

        Note: zero_flag keys should match targets list.

    Returns
    -------
    None

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
            zero_flag={'CCO': False,
                       'Vmca': False})
        outputs_list.append(output)

    d = {"Errors": {}, "Outputs": {}}

    d['Errors']['Average'] = np.nanmean([o[0]['TOTAL'] for o in outputs_list])
    for target in targets:
        d['Errors'][target] = np.nanmean([o[0][target] for o in outputs_list])
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
            bbox_to_anchor=(1.25, -0.5))
    return None
