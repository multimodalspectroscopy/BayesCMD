import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from math import log10, floor

sns.set_context('talk')
sns.set_style('ticks')


def round_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)



pfile = 'concatenated_results_190917.csv'
params=[r'r_t',r'r_0',r'r_m',r'cytox_tot_tis',r'Vol_mit',r'O2_n',r'v_cn',r'sigma_coll']
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

    accepted_limit = int(len(sorted_df)*frac/100)
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
        for t, l in zip(g.fig.get_children()[-1].texts, new_labels):
            t.set_text(l)
        lgd = g.fig.get_children()[-1]
        for i in range(2):
            lgd.legendHandles[i].set_sizes([50])
        
            
        return g



def infer_cmap(color):  
    if color == (0., 0., 1.):
        return 'Blues'
    elif color == (0., 0.5, 0.):
        return 'Greens'
    elif color == (1., 0., 0.):
        return 'Reds'
    elif color == (0.75, 0., 0.75):
        return 'Purples'

def kde_hue(x, y, **kws):
    ax = plt.gca()
    cmap = infer_cmap(kws['color'])
    sns.kdeplot(data=x, data2=y, ax=ax, cmap=cmap, n_levels=5, **kws)
    return ax

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

    accepted_limit = int(len(sorted_df) * frac / 100)

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



import sys
sys.path.append('../../..')
from bayescmd.util import findBaseDir

BASEDIR = os.path.abspath(findBaseDir('BayesCMD'))
os.environ['BASEDIR']='BayesCMD'





from bayescmd.bcmdModel import ModelBCMD




input_path = os.path.join(BASEDIR,'data','hx01.csv')
openopt_path = os.path.join(BASEDIR,'data','model_run_output.csv')




true_data = pd.read_csv(input_path)
openopt_data = pd.read_csv(openopt_path)




outputs = ['Vmca','CCO']
model_name = 'BS'
inputs = ['Pa_CO2', 'P_a', 'SaO2sup']
times = true_data['t'].as_matrix()




from bayescmd import abc
d0 = abc.import_actual_data(input_path)
input_data = abc.inputParse(d0, inputs)




def run_model(model):
    model.create_initialised_input()
    model.run_from_buffer()
    output = model.output_parse()
    return output




def plot_output(posterior_sample, param_names, plot=True):

    model = ModelBCMD(model_name,
                     inputs=input_data,
                     params=p,
                     times=times,
                     outputs=outputs)

    output = run_model(model)

    dist = abc.get_distance(d0,
                            output,
                            outputs,
                            distance='euclidean',
                            zero_flag=None)
    try:
        out_df = pd.DataFrame(data=output)
    except ValueError as e:
        print("error running for {}".format(posterior_sample[1]))
        raise e
    for k, v in dist.items():
        p[k] = v
    if plot:
        plt.rc('text', usetex=False)
        fig, ax = plt.subplots(2)
        with sns.plotting_context("talk", rc={"figure.figsize":(12,9)}):
            ax[0].plot(times, out_df['CCO'], 'b', label='Bayesian')
            ax[0].plot(times, true_data['CCO'], 'g', label='True')
            ax[0].plot(times, openopt_data['CCO'], 'r', label='openopt')
            t = 'CCO: $Error={CCO:.4f}$'
            p_str = ""
            i = 0
            for k in p.keys():
                if k not in dist.keys():
                    p_str+= "{0}={{{0}}}".format(k)
                    if (i+1)%3==0:
                        p_str+='\n'
                    else:
                        p_str+='    '
                    i+=1
            p_str=p_str.format(**p)
            ax[0].set_title(t.format(**p))
            ax[0].title.set_fontsize(19)
            ax[0].set_xlabel('Time (sec)')
            ax[0].set_ylabel(r'CCO ($\mu M$)')
            for item in ([ax[0].xaxis.label, ax[0].yaxis.label] +
             ax[0].get_xticklabels() + ax[0].get_yticklabels()):
                item.set_fontsize(17)
            
            ax[1].plot(times, out_df['Vmca'], 'b', label='Bayesian')
            ax[1].plot(times, true_data['Vmca'], 'g', label='True')
            ax[1].plot(times, openopt_data['Vmca'], 'r', label='openopt')
            t = 'Vmca: $Error={Vmca:.4f}$'
            ax[1].set_title(t.format(**p))
            ax[1].legend(prop={"size":17},bbox_to_anchor=(1.25, 1.5))
            ax[1].title.set_fontsize(19)
            ax[1].set_xlabel('Time (sec)')
            ax[1].set_ylabel(r'Vmca ($cm\,s^{-1}$)')
            plt.subplots_adjust(top=0.6)
            fig.suptitle("Top {}% of data".format(posterior_sample[0]*100))
            for item in ([ax[1].xaxis.label, ax[1].yaxis.label] +
             ax[1].get_xticklabels() + ax[1].get_yticklabels()):
                item.set_fontsize(17)
            plt.rc('text', usetex=False)
            plt.figtext(0.25,-0.05, p_str)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return p, output


_ = plot_output(list(posterior.items())[0], params)


# In[ ]:


import matplotlib.lines as mlines

def plot_repeated_outputs(n_repeats, frac):
    """
    inputs
    =======
    frac: Fraction to plot as a percentage
    """
    outputs_list= []
    while len(outputs_list)<n_repeats:
        p = result.iloc[:int(num_lines*(frac/100))][params].as_matrix()
        ps = stats.gaussian_kde(p.T).resample(1)
        try:
            p, o = plot_output((frac/100, ps), params, plot=False)
        except (ValueError, CalledProcessError, ModelRunError, TimeoutExpired):
            continue
        outputs_list.append((p, o))

    av_error = np.nanmean([o[0]['TOTAL'] for o in outputs_list])
    av_CCO_error = np.nanmean([o[0]['CCO'] for o in outputs_list])
    av_Vmca_error = np.nanmean([o[0]['Vmca'] for o in outputs_list])
    CCO_data = [o[1]['CCO'] for o in outputs_list]
    Vmca_data = [o[1]['Vmca'] for o in outputs_list]
    
    with sns.plotting_context("talk", rc={"figure.figsize":(12,9)}):
        fig, ax = plt.subplots(2)
        g = sns.tsplot(data=CCO_data, time=times, estimator=np.nanmean, ax=ax[0])
        true_plot, = ax[0].plot(times, true_data['CCO'], 'g', label='True')
        openopt_plot, = ax[0].plot(times, openopt_data['CCO'], 'r', label='Openopt')
        bayes_line = mlines.Line2D([], [], color=sns.color_palette()[0], label='Bayes')
        #ax[0].legend(handles=[bayes_line, true_plot], prop={"size":17})
        ax[0].set_title("CCO: Average Euclidean Distance of {:.4f}".format(av_CCO_error))
        ax[0].set_ylabel(r'CCO ($\mu M$)')
        ax[0].set_xlabel('Time (sec)')
        ax[0].title.set_fontsize(19)
        for item in ([ax[0].xaxis.label, ax[0].yaxis.label] +
         ax[0].get_xticklabels() + ax[0].get_yticklabels()):
            item.set_fontsize(17)
        ax[1].legend(handles=[bayes_line, true_plot, openopt_plot], prop={"size":17},bbox_to_anchor=(1.25, 1.5))
        
        g = sns.tsplot(data=Vmca_data, time=times, estimator=np.nanmean, ax=ax[1])
        true_plot, = ax[1].plot(times, true_data['Vmca'], 'g', label='True')
        openopt_plot, = ax[1].plot(times, openopt_data['Vmca'], 'r', label='Openopt')

        ax[1].set_title("Vmca: Average Euclidean Distance of {:.4f}".format(av_Vmca_error))
        ax[1].set_ylabel(r'CCO ($\mu M$)')
        ax[1].set_xlabel('Time (sec)')
        ax[1].title.set_fontsize(19)
        for item in ([ax[1].xaxis.label, ax[1].yaxis.label] +
         ax[1].get_xticklabels() + ax[1].get_yticklabels()):
            item.set_fontsize(17)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Simulated output for {} repeats using top {}% of data".format(n_repeats, frac))
    return None





plot_repeated_outputs(100, 0.01)

plot_repeated_outputs(100, 0.1)

plot_repeated_outputs(100, 1)




outputs_list= []
frac = 0.01
for i in range(50):
    p = result.iloc[:int(num_lines*(frac/100))][params].as_matrix()
    ps = stats.gaussian_kde(p.T).resample(1)
    outputs_list.append(plot_output((frac, ps), params, plot=False))




CCO_data = [o[1]['CCO'] for o in outputs_list]


Vmca_data = [o[1]['Vmca'] for o in outputs_list]



ax1 = sns.tsplot(data=CCO_data, time=times, err_style=["ci_band"], ci=[68,95])
ax1.plot(times, true_data['CCO'], 'g', label='True')
ax1.plot(times, openopt_data['CCO'], 'r', label='openopt')



ax2 = sns.tsplot(data=Vmca_data, time=times, err_style="ci_band", ci=[68,95], estimator=np.nanmean)
ax2.plot(times, openopt_data['Vmca'], 'r', label='openopt')
ax2.plot(times, true_data['Vmca'], 'g', label='True')
ax2.legend()




ax2 = sns.tsplot(data=CCO_data, time=times, err_style=["unit_traces", "ci_band"], ci=[100], estimator=np.nanmean)



rt_range = np.linspace(0.01,0.03,100)
rm_range = np.linspace(0.01, 0.04, 100)
r0_range = np.linspace(0.007, 0.0175, 100)
volmit_range = np.linspace(0.02, 0.12, 100)
cytox_range =  np.linspace(0.0025, 0.009, 100)

ranges = [rt_range, rm_range, r0_range, volmit_range, cytox_range]


