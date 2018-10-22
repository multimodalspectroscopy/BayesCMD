""" Generate various steady state data sets."""
from run_steadystate import RunSteadyState
import os
import distutils
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib as mpl
from PIL import Image
from io import BytesIO
mpl.rc('figure', dpi=400)


inputs = {"P_a": (40, 150), "Pa_CO2": (8, 80), "SaO2sup": (0.5, 1.0)}
title_dict = {"P_a": "Arterial Blood Pressure (mmHg)",
              "Pa_CO2": "Partial Pressure of $CO_2$ (mmHg)",
              "SaO2sup": "Arterial Oxygen Saturation (%)",
              "CBF": "$CBF/CBF_n$",
              "CMRO2": "$CMRO_2 (mMs^{-1})$",
              "HbT": "HbT ($\mu M$)",
              "CCO": "CCO ($\mu M$)"}
outputs = ["CMRO2", "CCO", "HbT", "CBF"]
r_ts = {"$r_t$: 0.018": 0.018,
"$r_t$: 0.016": 0.016, "$r_t$: 0.013":0.013, "$r_t$: 0.010": 0.01}

cbar = sns.color_palette("muted", n_colors=4)
direction = "both"

def TIFF_exporter(fig, fname, fig_dir = '.'):
    """
    Parameters
    ----------
    fig: matplotlib figure
    """
    
    # save figure
    # (1) save the image in memory in PNG format
    png1 = BytesIO()
    fig.savefig(png1, format='png', bbox_inches='tight')

    # (2) load this image into PIL
    png2 = Image.open(png1)

    # (3) save as TIFF
    png2.save(os.path.join(fig_dir,'{}.tiff'.format(fname)))
    png1.close()
    return True


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow as a percentile.
                If None, mean of xdata is taken.
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    else:
        position = np.percentile(xdata, position)
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
        arrowstyle = "->"
    else:
        end_ind = start_ind - 1
        arrowstyle = "<-"

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle=arrowstyle, color=color),
                       size=size
                       )


for o in outputs:
    print("Running steady state - {}".format(o))
    base_work_dir = os.path.expanduser(os.path.join('~',
                                                    'Dropbox',
                                                    'phd',
                                                    'PLOS_paper',
                                                    'muscular_tension',
                                                    'Figures',
                                                    o))
    distutils.dir_util.mkpath(base_work_dir)
    for i, r in inputs.items():
        data = {}

        print("\tRunning steady state - {}".format(i))
        for k, v  in r_ts.items():
            workdir = os.path.join(base_work_dir, k)
            distutils.dir_util.mkpath(workdir)
            print("\t\tRunning r_t {}".format(k))
            config = {
                "model_name": "BS",
                "inputs": i,
                "parameters": {
                    "r_t": v
                },
                "targets": [o],
                "max_val": r[1],
                "min_val": r[0],
                "debug": True,
                "direction": direction
            }

            model = RunSteadyState(conf=config, workdir=workdir)
            output = model.run_steady_state()
            data[k] = output
            with open(os.path.join(workdir,
                                   "{}_{}.json".format(i, direction)),
                      'w') as f:
                json.dump(data, f)

        fig, ax = plt.subplots()

        for idx, k in enumerate(r_ts.keys()):
            if i == "SaO2sup":
                data[k][i] = [x*100 for x in data[k][i]]

            if o == "CBF":
                data[k][o] = [x/0.0125 for x in data[k][o]]
            ax.set_position([0.1, 0.1, 0.7, 0.8])
            line = ax.plot(data[k][i][:len(data[k][i]) // 2 + 1],
                           data[k][o][:len(data[k][o]) // 2 + 1],
                           label=k, color=cbar[idx])[0]
            add_arrow(line, position=60, size=24)
            line = ax.plot(data[k][i][len(data[k][i]) // 2:],
                           data[k][o][len(data[k][o]) // 2:],
                           color=cbar[idx], linestyle='--')[0]
            add_arrow(line, direction='left', position=25, size=24)

        # ax.set_title("Steady state for varying levels\nof "
        #              "$r_t$",
        #              size=20)
        ax.set_ylabel(title_dict[o], size=16)
        ax.set_xlabel(title_dict[i], size=16)
        ax.tick_params(axis='both', which='major', labelsize=16)

        legend = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                           prop={'size': 22})

        # fig.savefig(os.path.join(base_work_dir, "{}_{}.png".format(i,
        #                                                            direction)),
        #             bbox_inches="tight")
        TIFF_exporter(fig, "{}_{}".format(i, direction), fig_dir=base_work_dir)
