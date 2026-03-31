import re
import numpy as np
from glob import glob
from collections import defaultdict
import etils.epath as epath
from matplotlib import pyplot as plt
import csv

_HERE = epath.Path(__file__).parent

fall_time_logs = glob(str(_HERE) + "/logs/*fall_times.csv")

fall_times = {}
for log in fall_time_logs:
    with open(log, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            command = row[0]
            fall_time = float(row[1])
            if command not in fall_times:
                fall_times[command] = []
            if fall_time == -1.0:
                fall_time = 30.0
            fall_times[command].append(fall_time)

plt.figure()
plt.violinplot(fall_times.values())
plt.xticks(range(len(fall_times)), fall_times.keys(), rotation=90)
plt.show()

# --- Velocity tracking separated by model, backlash, and run ---

velocity_logs = glob(str(_HERE) + "/logs/*velocities.csv")

FILENAME_RE = re.compile(
    r"(\d+)_(.+\.onnx)_bl([\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_velocities\.csv"
)

# data[model][backlash] -> {cmd_x, cmd_y, cmd_z, real_x, real_y, real_z}
data = defaultdict(lambda: defaultdict(lambda: {
    "cmd_x": [], "cmd_y": [], "cmd_z": [],
    "real_x": [], "real_y": [], "real_z": [],
}))

for log in velocity_logs:
    fname = log.split("/")[-1]
    m = FILENAME_RE.match(fname)
    if not m:
        print(f"Skipping unrecognized file: {fname}")
        continue

    run_id = m.group(1)
    model_name = m.group(2)
    backlash = float(m.group(3))
    cmd_x, cmd_y, cmd_z = float(m.group(4)), float(m.group(5)), float(m.group(6))

    with open(log, "r") as f:
        rows = list(csv.reader(f))
        if len(rows) <= 501:
            continue
        velocities = np.array(
            [[float(r[0]), float(r[1]), float(r[5])] for r in rows[500:]]
        )
        mean_vel = np.mean(velocities, axis=0)

    entry = data[model_name][backlash]
    entry["cmd_x"].append(cmd_x)
    entry["cmd_y"].append(cmd_y)
    entry["cmd_z"].append(cmd_z)
    entry["real_x"].append(mean_vel[0])
    entry["real_y"].append(mean_vel[1])
    entry["real_z"].append(mean_vel[2])

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
                unique_cmds, mean_reals, "o",
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