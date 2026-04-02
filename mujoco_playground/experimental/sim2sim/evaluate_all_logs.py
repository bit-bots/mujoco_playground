import pickle
import re
import numpy as np
from glob import glob
from collections import defaultdict
import etils.epath as epath
from matplotlib import pyplot as plt
import csv
import copy

from tqdm import tqdm

_HERE = epath.Path(__file__).parent
MAX_TIME = 10.0

fall_time_logs = glob(str(_HERE) + "/logs/*fall_times.csv")

fall_times = {"zero": {"0.0": {}, "0.01": {}, "0.025": {}, "0.05": {}, "0.075": {}, "0.1": {}}, 
              "rand": {"0.0": {}, "0.01": {}, "0.025": {}, "0.05": {}, "0.075": {}, "0.1": {}}}

# 007_wolfgang_grc_zero_bl.onnx_bl0.025_fall_times.csv
fall_times_regex = r"(\d+)_wolfgang_grc_(zero|rand)_bl.onnx_bl([\d.]+)_fall_times.csv"

for logfile in fall_time_logs:
    fname = logfile.split("/")[-1]
    m = re.match(fall_times_regex, fname)
    if not m:
        print(f"Skipping unrecognized file: {fname}")
        continue

    run_id = m.group(1)
    model_name = m.group(2)
    backlash = m.group(3)
    with open(logfile, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            command = row[0]
            fall_time = float(row[1])
            if command not in fall_times[model_name][backlash]:
                fall_times[model_name][backlash][command] = []
            fall_times[model_name][backlash][command].append(fall_time)


# count number of falls for each backlash, model, velocity combination
fall_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for model_name in fall_times.keys():
    for backlash in fall_times[model_name].keys():
        for command in fall_times[model_name][backlash].keys():
            fallen_times = 0
            for fall_time in fall_times[model_name][backlash][command]:
                if fall_time != -1.0:
                    fallen_times += 1
            fall_counts[model_name][backlash][command] = fallen_times

# y axis number of falls,
# x axis velocity commands
# one color per model, shades for backlash values

color_without_backlash ='#fe6100'
color_with_backlash = '#785ef0'


model_cmap = {"rand": plt.cm.Blues, "zero": plt.cm.Reds}

fig, ax = plt.subplots(2,3)
fig.set_size_inches(30, 20)
# reduce outer margins
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
for model_name in ["rand", "zero"]:
    color = color_with_backlash if model_name == "rand" else color_without_backlash
    for bl_idx, backlash in enumerate(["0.0", "0.01", "0.025", "0.05", "0.075", "0.1"]):
        ax[bl_idx%2][bl_idx//2].set_title(f"Backlash {backlash}")
        ax[bl_idx%2][bl_idx//2].set_xlabel("Velocity Command")
        ax[bl_idx%2][bl_idx//2].set_ylabel("Number of Falls")
        ax[bl_idx%2][bl_idx//2].set_ylim(0, 100)
        # set x tick rotation:
        ax[bl_idx%2][bl_idx//2].tick_params(axis='x', labelrotation=75)
        
        x_pos = list(range(len(fall_counts[model_name][backlash].keys())))
        if model_name == "rand":
            x_pos = [x + 0.2 for x in x_pos]
        else:
            x_pos = [x - 0.2 for x in x_pos]
        labels = ["(" + cmd.replace("_", ",") + ")" for cmd in fall_counts[model_name][backlash].keys()]
        ax[bl_idx%2][bl_idx//2].bar(x_pos,
                                    list(fall_counts[model_name][backlash].values()),
                                    0.35,
                                    color=color,
                                    alpha=0.7,
                                    tick_label=labels)

plt.legend()
plt.show()

# --- Velocity tracking separated by model, backlash, and run ---

#match only the first 20 logs 
velocity_logs = sorted(glob(str(_HERE) + "/logs/*velocities.csv"))

FILENAME_RE = re.compile(
    r"(\d+)_(.+\.onnx)_bl([\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_velocities\.csv"
)

# data[model][backlash] -> {cmd_x, cmd_y, cmd_z, real_x, real_y, real_z}
bl_values = ["0.0", "0.01", "0.025", "0.05", "0.075", "0.1"]
models = ["wolfgang_grc_rand_bl.onnx", "wolfgang_grc_zero_bl.onnx"]

data = {}
for model in models:
    data[model] = {}
    for bl_value in bl_values:
        data[model][bl_value] = {
            "cmd_x": [], "cmd_y": [], "cmd_z": [],
            "real_x": [], "real_y": [], "real_z": [],
        }

# check if pickle exists:
if (_HERE / "velocities.pkl").exists():
    with open(_HERE / "velocities.pkl", "rb") as f:
        data = pickle.load(f)
else:
    for log in tqdm(velocity_logs):
        fname = log.split("/")[-1]
        m = FILENAME_RE.match(fname)
        if not m:
            print(f"Skipping unrecognized file: {fname}")
            continue

        run_id = m.group(1)
        model_name = m.group(2)
        backlash = m.group(3)
        cmd_x, cmd_y, cmd_z = float(m.group(4)), float(m.group(5)), float(m.group(6))

        # filter out if fallen
        #fallen_log_filename = f"{run_id}_wolfgang_grc_{model_name}_bl{backlash}_fall_times.csv"
        #fallen_log_path = _HERE / "logs" / fallen_log_filename
        #if fallen_log_path.exists():
        #    with open(fallen_log_path, "r") as f:
        #        rows = list(csv.reader(f))
        #        


        with open(log, "r") as f:
            rows = list(csv.reader(f))
            if len(rows) <= 1502:
                continue
            velocities = np.array(
                [[float(r[0]), float(r[1]), float(r[5])] for r in rows[500:-1000]]
            )
            mean_vel = np.mean(velocities, axis=0)

        entry = data[model_name][backlash]
        entry["cmd_x"].append(cmd_x)
        entry["cmd_y"].append(cmd_y)
        entry["cmd_z"].append(cmd_z)
        entry["real_x"].append(mean_vel[0])
        entry["real_y"].append(mean_vel[1])
        entry["real_z"].append(mean_vel[2])

    with open(_HERE / "velocities.pkl", "wb") as f:
        pickle.dump(dict(data), f)


# --- Plotting: one color per model, shades for backlash values ---

model_names = sorted(data.keys())
backlash_values = sorted({bl for md in data.values() for bl in md})
n_bl = len(backlash_values)

BASE_CMAPS = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]
model_cmap = {name: BASE_CMAPS[i % len(BASE_CMAPS)] for i, name in enumerate(model_names)}

AXES = [
    ("cmd_x", "real_x", "Linear Velocity X"),
    ("cmd_y", "real_y", "Linear Velocity Y"),
    ("cmd_z", "real_z", "Angular Velocity Z"),
]

for cmd_key, real_key, title in AXES:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Commanded vs Achieved {title}")
    ax.set_xlabel("Commanded")
    ax.set_ylabel("Achieved")

    for model in model_names:
        cmap = model_cmap[model]
        for bl_idx, bl_val in enumerate(backlash_values):
            if bl_val not in data[model]:
                continue
            color = cmap(0.4 + 0.6 * bl_idx / max(n_bl - 1, 1))
            e = data[model][bl_val]
            cmd = np.array(e[cmd_key])
            real = np.array(e[real_key])
            mask = cmd != 0.0
            if not np.any(mask):
                continue
            cmd_masked = cmd[mask]
            real_masked = real[mask]
            unique_cmds = np.unique(cmd_masked)
            mean_reals = np.array(
                [np.mean(real_masked[cmd_masked == c]) for c in unique_cmds]
            )
            ax.plot(
                unique_cmds, mean_reals, "o-",
                color=color, label=f"{model} bl={bl_val}", markersize=4, alpha=0.8,
            )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize="small", loc="upper left")
    plt.tight_layout()
    plt.show()