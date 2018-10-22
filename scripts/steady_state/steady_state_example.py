""" Generate various steady state data sets."""
from run_steadystate import RunSteadyState
import os
import distutils
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


inputs = {"P_a": (30, 70), "Pa_CO2": (8, 160), "SaO2sup": (0.2, 1.0)}

temperatures = [37, 35, 33.5]

cbar = sns.color_palette("muted", n_colors=4)

# Define global work directory
WORKDIR = ""

# Set steady state direction - "up", "down" or "both"
for direction in ["up", "down"]:
    for i, r in inputs.items():
        data = {}
        workdir = os.path.join(WORKDIR, direction)
        distutils.dir_util.mkpath(workdir)
        print("Running steady state - {}".format(i))
        # Run steady state for different model states e.g. temperatures
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
                "debug": True,
                "direction": direction
            }

            model = RunSteadyState(conf=config, workdir=workdir)
            output = model.run_steady_state()
            data[t] = output
            with open(os.path.join(workdir,
                                   "{}_{}.json".format(i, direction)), 'w') as f:
                json.dump(data, f)

        fig, ax = plt.subplots()
        for idx, t in enumerate(temperatures):
            ax.plot(data[t][i], data[t]['CBF'], label=t, color=cbar[idx])

        ax.set_title("Steady state for varying {} - {}".format(i, direction))
        ax.set_ylabel("CBF")
        ax.set_xlabel(i)
        legend = ax.legend(loc='upper center')

        fig.savefig("directory/to/store/figures/Figures/{}_{}"
                    ".png".format(i, direction),
                    bbox_inches="tight")

