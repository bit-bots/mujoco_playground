import argparse
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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 20,
})

parser = argparse.ArgumentParser()
parser.add_argument("--redo_cache", action="store_true", help="Recompute and overwrite the cached velocities.pkl")
args = parser.parse_args()

_HERE = epath.Path(__file__).parent
MAX_TIME = 10.0

fall_time_logs = glob(str(_HERE) + "/logs/*fall_times.csv")

bl_values = ["0.0", "0.0125", "0.025", "0.0375", "0.05", "0.0625", "0.075", "0.0875", "0.1"]

fall_times = {
    "zero": {bl: {} for bl in bl_values},
    "rand": {bl: {} for bl in bl_values},
}
models = ["wolfgang_grc_rand_bl.onnx", "wolfgang_grc_zero_bl.onnx"]


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

CMD_ORDER = [
    "0.00_0.00_0.00",
    "0.20_0.00_0.00", "0.40_0.00_0.00",
    "-0.20_0.00_0.00", "-0.40_0.00_0.00",
    "0.00_0.20_0.00", "0.00_0.40_0.00",
    "0.00_-0.20_0.00", "0.00_-0.40_0.00",
    "0.00_0.00_0.50", "0.00_0.00_1.00", "0.00_0.00_1.50",
    "0.00_0.00_-0.50", "0.00_0.00_-1.00", "0.00_0.00_-1.50",
]


def _cmd_label(cmd_str):
    x, y, z = [float(v) for v in cmd_str.split("_")]
    if x == 0.0 and y == 0.0 and z == 0.0:
        return r"$|v|=0$"
    if x != 0.0:
        return rf"$v_x={x:g}$"
    if y != 0.0:
        return rf"$v_y={y:g}$"
    return rf"$v_{{\theta}}={z:g}$"


color_without_backlash = '#fe6100'
color_with_backlash = '#648fff'

plot_bl_values = [bl for bl in bl_values if float(bl) > 0.0375]

# 2 on top (left-aligned), legend to the right, 3 on bottom
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 6, hspace=0.65, wspace=0.08,
                      left=0.06, right=0.98, top=0.95, bottom=0.16)
axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[1, 0:2]),
    fig.add_subplot(gs[1, 2:4]),
    fig.add_subplot(gs[1, 4:6]),
]
legend_ax = fig.add_subplot(gs[0, 4:6])
legend_ax.axis('off')

for model_name in ["rand", "zero"]:
    color = color_with_backlash if model_name == "rand" else color_without_backlash
    label = "with backlash in training" if model_name == "rand" else "without backlash in training"
    for bl_idx, backlash in enumerate(plot_bl_values):
        a = axes[bl_idx]
        a.set_title(f"bl = {backlash}")
        if bl_idx in (0, 2):
            a.set_ylabel("Number of Falls")
        else:
            a.tick_params(labelleft=False)
        a.set_ylim(0, 500)
        a.tick_params(axis='x', labelrotation=75)

        cmds = [c for c in CMD_ORDER if c in fall_counts[model_name][backlash]]
        x_pos = [i + (0.2 if model_name == "rand" else -0.2) for i in range(len(cmds))]
        labels = [_cmd_label(cmd) for cmd in cmds]
        a.bar(x_pos, [fall_counts[model_name][backlash][c] for c in cmds], 0.35,
              color=color, alpha=0.7,
              label=label if bl_idx == 0 else None,
              tick_label=labels)

handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc='center left')
plt.savefig(_HERE / "fall_counts.pdf")
plt.close(fig)

# --- Velocity tracking separated by model, backlash, and run ---

velocity_logs = sorted(glob(str(_HERE) + "/logs/*velocities.npy"))

FILENAME_RE = re.compile(
    r"(\d+)_(.+\.onnx)_bl([\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_velocities\.npy"
)

# data[model][backlash] -> {cmd_x, cmd_y, cmd_z, real_x, real_y, real_z, n_samples}
data = {}
for model in models:
    data[model] = {}
    for bl_value in bl_values:
        data[model][bl_value] = {
            "cmd_x": [], "cmd_y": [], "cmd_z": [],
            "real_x": [], "real_y": [], "real_z": [],
            "n_samples": [],
        }

# check if pickle exists:
if not args.redo_cache and (_HERE / "logs" / "velocities.pkl").exists():
    with open(_HERE / "logs" / "velocities.pkl", "rb") as f:
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


        arr = np.load(log)
        if len(arr) <= 2500:
            continue
        velocities = arr[1250:-1250, [0, 1, 5]]
        n_samples = len(velocities)
        mean_vel = np.mean(velocities, axis=0)

        entry = data[model_name][backlash]
        entry["cmd_x"].append(cmd_x)
        entry["cmd_y"].append(cmd_y)
        entry["cmd_z"].append(cmd_z)
        entry["real_x"].append(mean_vel[0])
        entry["real_y"].append(mean_vel[1])
        entry["real_z"].append(mean_vel[2])
        entry["n_samples"].append(n_samples)

    with open(_HERE / "logs" / "velocities.pkl", "wb") as f:
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

plt.rcParams.update({"font.size": 18})
fig, axes_2x2 = plt.subplots(2, 2, figsize=(12, 10))
ax_positions = {
    "cmd_x": axes_2x2[0, 0],
    "cmd_y": axes_2x2[0, 1],
    "cmd_z": axes_2x2[1, 0],
}
legend_ax = axes_2x2[1, 1]
legend_ax.axis("off")

all_handles, all_labels = [], []

for cmd_key, real_key, title in AXES:
    ax = ax_positions[cmd_key]
    ax.set_title(title)
    if cmd_key == "cmd_x" or cmd_key == "cmd_y":
        ax.set_xlabel("Commanded $\\mathrm{m/s}$")
    elif cmd_key == "cmd_z":
        ax.set_xlabel("Commanded $\\mathrm{rad/s}$") 
    if cmd_key == "cmd_x" or cmd_key == "cmd_z":
        ax.set_ylabel("Achieved $\\mathrm{m/s}$" if cmd_key == "cmd_x" else "Achieved $\\mathrm{rad/s}$")
    for model in model_names:
        cmap = model_cmap[model]
        for bl_idx, bl_val in enumerate(backlash_values):
            if bl_val not in data[model]:
                continue
            color = cmap(0.4 + 0.6 * bl_idx / max(n_bl - 1, 1))
            e = data[model][bl_val]
            cmd = np.array(e[cmd_key])
            real = np.array(e[real_key])
            n = np.array(e["n_samples"])
            all_zero = (
                (np.array(e["cmd_x"]) == 0.0) &
                (np.array(e["cmd_y"]) == 0.0) &
                (np.array(e["cmd_z"]) == 0.0)
            )
            mask = (cmd != 0.0) | all_zero
            if not np.any(mask):
                continue
            cmd_masked = cmd[mask]
            real_masked = real[mask]
            n_masked = n[mask]
            unique_cmds = np.unique(cmd_masked)
            mean_reals = np.array(
                [np.sum(real_masked[cmd_masked == c] * n_masked[cmd_masked == c]) / np.sum(n_masked[cmd_masked == c])
                 for c in unique_cmds]
            )
            if "rand" in model:
                label_str = "backlash in training"
            else:
                label_str = f"no backlash in training; bl={bl_val}"
            line, = ax.plot(
                unique_cmds, mean_reals, "o-",
                color=color, label=label_str, markersize=4, alpha=0.8,
            )
            if cmd_key == "cmd_x":
                all_handles.append(line)
                all_labels.append(label_str)
    data_min = ax.get_ylim()[0]
    data_max = ax.get_ylim()[1]
    ax.plot([data_min, data_max], [data_min, data_max], "k--", alpha=0.4, linewidth=1)

legend_ax.legend(all_handles, all_labels, loc="center", fontsize=15, ncol=2,
                 borderpad=0.2, labelspacing=0.2, handlelength=1.0,
                 handleheight=0.35, handletextpad=0.4, borderaxespad=0.25,
                 columnspacing=0.5)
plt.tight_layout(pad=0.5)
plt.savefig(_HERE / "cmd_vs_real_velocities.pdf")
plt.close(fig)