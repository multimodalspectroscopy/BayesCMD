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

for direction in ["up", "down"]:
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

        fig.savefig("/home/buck06191/Dropbox/phd/hypothermia/Figures/{}_{}"
                    ".png".format(i, direction),
                    bbox_inches="tight")

outputs = ["CMRO2", "CCO", "DHbT", "CBF", "DHbdiff",
           "TOI", "Vmca", "DHbO2", "DHHb", "HHb", "HbO2"]

# For debugging
outputs.extend(["k_MAshut", "k_nMAshut", "Q_temp", "_ADP", "_ATP"])

q10_range = [1, 1.5, 2, 2.5, 3, 3.5,  4, 4.5, 5]

cbar = sns.color_palette("Set1", n_colors=len(q10_range))

data = {}
direction = "both"
workdir = os.path.join('.', 'build', 'steady_state', 'bp_hypothermia',
                       "q10")
distutils.dir_util.mkpath(workdir)

for q in q10_range:
    print("Running Q10 {}".format(q))
    config = {
        "model_name": "bp_hypothermia",
        "inputs": "temp",
        "parameters": {
            "Q_10": q
        },
        "targets": outputs,
        "max_val": 37,
        "min_val": 31,
        "debug": False,
        "direction": direction
    }

    model = RunSteadyState(conf=config, workdir=workdir)
    output = model.run_steady_state()
    data[q] = output
now = datetime.now().strftime('%d%m%yT%H%M')
with open(os.path.join(workdir, "{}.json".format(now)), 'w') as f:
    json.dump(data, f)

for o in outputs:
    if direction == "both":
        q = 2.5
        fig, ax = plt.subplots()
        ax.plot(data[q]["temp"][:len(data[q][o]) // 2 + 1],
                data[q][o][:len(data[q][o]) // 2 + 1],
                label="Up")
        ax.plot(data[q]["temp"][len(data[q][o]) // 2:],
                data[q][o][len(data[q][o]) // 2:],
                label="Down")

        ax.set_title("Steady state for {}".format(o))
        ax.set_ylabel(o)
        ax.set_xlabel("Temp (C)")
        ax.legend()

        path = "/home/buck06191/Dropbox/phd/hypothermia/Figures/varying_q10/{}".format(
            now)
        distutils.dir_util.mkpath(path)
        fig.savefig(os.path.join(path, "{}_both.png".format(o)),
                    bbox_inches="tight")
        plt.close()

    else:
        fig, ax = plt.subplots()
        for idx, q in enumerate(q10_range):
            # if direction == "both":
            #     ax.plot(output["temp"][:len(output[o]) // 2 + 1],
            #             output[o][:len(output[o]) // 2 + 1],
            #             label="Up")
            #     ax.plot(output["temp"][len(output[o]) // 2:],
            #             output[o][len(output[o]) // 2:],
            #             label="Down")
            # else:
            ax.plot(data[q]["temp"],
                    data[q][o], label=q, color=cbar[idx])
        ax.set_title("{} response to temperature".format(o))
        ax.set_ylabel(o)
        ax.set_xlabel("Temp (C)")
        ax.legend()

        path = "/home/buck06191/Dropbox/phd/hypothermia/Figures/varying_q10/{}".format(
            now)
        distutils.dir_util.mkpath(path)
        fig.savefig(os.path.join(path, "{}.png".format(o)),
                    bbox_inches="tight")
        plt.close()
