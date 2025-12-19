import csv
from absl import app
from absl import flags
import matplotlib.pyplot as plt
from etils import epath
import numpy as np

_HERE = epath.Path(__file__).parent
_ONNX_MODEL = flags.DEFINE_string(
    "onnx_model", "wolfgang_policy.onnx", "Name of the ONNX model file"
)
_BACKLASH = flags.DEFINE_float(
    "backlash", 0.05, "Backlash value for backlash joints (in radians)"
)

def main(_):
    commands = [
        (0.5, 0.0, 0.0),  # forward
        (0.0, 0.5, 0.0),  # left
        (0.0, -0.5, 0.0),  # right
        (0.0, 0.0, 0.5),  # turn left
        (0.0, 0.0, -0.5),  # turn right
    ]

    for command in commands:
        cmd_str = "_".join([f"{c:.2f}" for c in command])
        with open(_HERE / "logs" / f"{_ONNX_MODEL.value}_bl{_BACKLASH.value}_{cmd_str}_velocities.csv", "r") as f:
            reader = csv.reader(f)
            velocities = np.array([[float(value) for value in row] for row in reader])
        plt.figure()
        plt.title(f"Command: {command}")
        # plot value smoothed by running average
        window_size = 500

        #plt.plot(velocities[:, 0], label="Vel X", linestyle='--', color='red')
        smoothed_vel_x = np.convolve(velocities[:, 0], np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_vel_x, label="Smoothed Vel X", color='red')


        #plt.plot(velocities[:, 5], label="Ang Vel Z", linestyle='--', color='blue')
        smoothed_ang_vel_z = np.convolve(velocities[:, 5], np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_ang_vel_z, label="Smoothed Ang Vel Z", color='blue')


        #plt.plot(velocities[:, 1], label="Vel Y", linestyle='--', color='green')
        smoothed_vel_y = np.convolve(velocities[:, 1], np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_vel_y, label="Smoothed Vel Y", color='green')

        plt.xlabel("Timestep")
        plt.ylabel("Velocity (m/s)")
        plt.plot(command[0] * np.ones_like(velocities[:, 0]), 'r--', label="Target Vel X")
        plt.plot(command[1] * np.ones_like(velocities[:, 1]), 'g--', label="Target Vel Y")
        plt.plot(command[2] * np.ones_like(velocities[:, 5]), 'b--', label="Target Ang Vel Z")
        plt.legend()
        plt.grid()
        plt.savefig(_HERE / "logs" / f"{_ONNX_MODEL.value}_{cmd_str}_velocities.png")
        plt.close()

if __name__ == "__main__":
    app.run(main)