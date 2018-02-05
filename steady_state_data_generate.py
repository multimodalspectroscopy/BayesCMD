""" Generate various steady state data sets."""
from run_steadystate import RunSteadyState
import os
import distutils
import json
import matplotlib.pyplot as plt
import seaborn as sns


inputs = {"P_a": (20, 150), "Pa_CO2": (8, 80), "SaO2sup": (0.2, 1.0)}

temperatures = [31.5, 33.5, 35, 37]

cbar = sns.color_palette("muted", n_colors=4)

for i, r in inputs.items():
    data = {}
    workdir = os.path.join('.', 'build', 'steady_state', 'bp_hypothermia')
    distutils.dir_util.mkpath(workdir)
    print("Running steady state - {}".format(i))
    for t in temperatures:
        print("Running temperature {}C".format(t))
        config = {
            "model_name": "bp_hypothermia",
            "inputs": i,
            "parameters": {
                "temp": t
            },
            "targets": ["CBF"],
            "max_val": r[1],
            "min_val": r[0],
            "debug": False
        }

        model = RunSteadyState(conf=config, workdir=workdir)
        output = model.run_steady_state()
        data[t] = output
    with open(os.path.join(workdir, "{}.json".format(i)), 'w') as f:
        json.dump(data, f)

    fig, ax = plt.subplots()
    for idx, t in enumerate(temperatures):
        ax.plot(data[t][i], data[t]['CBF'], label=t, color=cbar[idx])

    ax.set_title("Steady state for varying {}".format(i))
    ax.set_ylabel("CBF")
    ax.set_xlabel(i)
    legend = ax.legend(loc='upper center')

    plt.show()
