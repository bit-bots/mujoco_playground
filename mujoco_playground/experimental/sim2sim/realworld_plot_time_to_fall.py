import matplotlib.pyplot as plt
import numpy as np
import etils.epath as epath


_HERE = epath.Path(__file__).parent


plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX to render all text
    "font.family": "serif",        # Use serif fonts
    "font.serif": ["Times"],       # Match IEEE Times New Roman style
    "pdf.fonttype": 42,            # Output Type 42 (TrueType) fonts
    "ps.fonttype": 42,             # Output Type 42 (TrueType) fonts
    "font.size": 8                 # Match typical caption/label sizes
})
# Data provided
data = {
    "(0.0, 0.0, 0.0)": [[30, 30, 30], [30, 30, 30]],

    "(0.2, 0.0, 0.0)": [[ 3,  4,  3], [25, 30, 30]],
    "(0.4, 0.0, 0.0)": [[ 2,  1,  3], [30, 30, 30]],

    "(-0.2, 0.0, 0.0)": [[30, 30, 30], [30, 30, 30]],
    "(-0.4, 0.0, 0.0)": [[30, 30, 30], [30, 30, 30]],

    "(0.0, 0.2, 0.0)": [[8, 6, 8], [30, 30, 7]],
    "(0.0, 0.4, 0.0)": [[4, 5, 2], [18, 20, 7]],
    "(0.0, -0.2, 0.0)": [[ 4, 30, 30], [30, 30, 30]],
    "(0.0, -0.4, 0.0)": [[30, 30, 30], [30, 30, 30]],

    "(0.0, 0.0, 0.5)": [[30, 30, 30], [30, 30, 30]],
    "(0.0, 0.0, 1.0)": [[30, 15, 13], [30, 30, 30]],
    "(0.0, 0.0, 1.5)": [[30, 30, 30], [30, 30, 30]],

    "(0.0, 0.0, -0.5)": [[30, 21, 30], [30, 30, 30]],
    "(0.0, 0.0, -1.0)": [[12, 30, 18], [30, 30, 30]],
    "(0.0, 0.0, -1.5)": [[30, 13, 30], [30, 30, 30]],
}

def _cmd_label(key):
    x, y, z = [float(v) for v in key.strip("()").split(",")]
    if x == 0.0 and y == 0.0 and z == 0.0:
        return r"$|v|=0$"
    if x != 0.0:
        return rf"$v_x={x:g}$"
    if y != 0.0:
        return rf"$v_y={y:g}$"
    return rf"$v_{{\theta}}={z:g}$"


# Extract labels and values
labels = [_cmd_label(k) for k in data.keys()]
mean1 = [np.mean(v[0]) for v in data.values()]
mean2 = [np.mean(v[1]) for v in data.values()]

# Setup positioning for grouped bars
x = np.arange(len(labels))
width = 0.35  # Increased from 0.25 to make bars wider and closer

# Create the plot
plt.figure(figsize=(14, 7))

# Plot bars with reduced spacing
bars1 = plt.bar(x - width/2, mean1, width, label='No Backlash', color='#fe6100', alpha=0.7)
bars2 = plt.bar(x + width/2, mean2, width, label='Randomized Backlash', color='#785ef0', alpha=0.7)

# Add individual data points
for i, (label, values) in enumerate(data.items()):
    # Plot individual points for condition 1 (No Backlash)
    x_pos1 = x[i] - width/2
    plt.scatter([x_pos1] * len(values[0]), values[0], color='darkred', s=60, alpha=0.8, zorder=3)
    
    # Plot individual points for condition 2 (Randomized Backlash)
    x_pos2 = x[i] + width/2
    plt.scatter([x_pos2] * len(values[1]), values[1], color='darkblue', s=60, alpha=0.8, zorder=3)

# Formatting labels and title
plt.ylabel('Time to Fall (s)', fontsize=24)
plt.title('Time to Fall vs Velocity Commands ($x$ (m/s), $y$ (m/s), $\\theta$ (rad/s))', fontsize=24)
plt.xticks(x, labels, rotation=60, ha='right', fontsize=24)
plt.legend(fontsize=20, loc='lower right')
plt.xlim(x[0] - 0.6, x[-1] + 0.6)

# Set y-axis to show full range
plt.ylim(0, 32)

# Add grid for better readability
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(_HERE / "realworld_time_to_fall.pdf")
plt.show()

