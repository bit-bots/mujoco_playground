"""Evaluate an ONNX policy with different velocity commands."""

from absl import app
from absl import flags
from absl import logging
from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
import time

from mujoco_playground._src.locomotion.wolfgang import wolfgang_constants
from mujoco_playground._src.locomotion.wolfgang.base import get_assets

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"

_BACKLASH = flags.DEFINE_float(
    "backlash", 0.02, "Backlash value for backlash joints (in radians)"
)
_RANDOM_BACKLASH = flags.DEFINE_boolean(
    "random_backlash", False, "Whether to randomize the backlash joint ranges"
)
_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed", 1, "Random seed"
)
_ONNX_MODEL = flags.DEFINE_string(
    "onnx_model", "wolfgang_policy_bl005.onnx", "Name of the ONNX model file"
)
_MAX_TIME = flags.DEFINE_float(
    "max_time", 30.0, "Maximum simulation time per test (in seconds)"
)
_VISUALIZE = flags.DEFINE_boolean(
    "visualize", False, "Whether to visualize the evaluation"
)
_REALTIME = flags.DEFINE_boolean(
    "realtime", False, "Whether to run the simulation in real-time"
)
_DISABLE_VELOCITY_LOG = flags.DEFINE_boolean(
    "disable_velocity_log", False, "Whether to disable the velocity log"
)

class OnnxController:
    """ONNX controller for the wolfgang humanoid."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        ctrl_dt: float,
        n_substeps: int,
        action_scale: float = 0.5,
    ):
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        self._phase = np.array([0.0, np.pi])
        self._gait_freq = 1.5
        self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt
        self._command = [0.0, 0.0, 0.0]

    def set_command(self, command):
        self._command = command

    def reset(self):
        self._last_action = np.zeros_like(self._default_angles, dtype=np.float32)
        self._counter = 0
        self._phase = np.array([0.0, np.pi])

    def get_obs(self, model, data) -> np.ndarray:
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])

        joint_angles = data.qpos[7::2] - self._default_angles
        joint_velocities = data.qvel[6::2]
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        command = np.array(self._command, dtype=np.float32)
        obs = np.hstack([
            gyro,
            gravity,
            command,
            joint_angles,
            joint_velocities,
            self._last_action,
            phase,
        ])
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_input = {"obs": obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
            phase_tp1 = self._phase + self._phase_dt
            self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


def run_experiment(command, policy, model, data, mj_viewer):
    #logging.info("Test command: %s", command)
    policy.set_command(command)
    policy.reset()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sim_time = 0.0
    last_sync_time = -1.0
    real_start_time = time.time()
    cmd_str = "_".join([f"{c:.2f}" for c in command])
    velocity_log_path = _HERE / ("dummy.csv" if _DISABLE_VELOCITY_LOG.value else f"logs/{_RANDOM_SEED.value:03d}_{_ONNX_MODEL.value}_bl{_BACKLASH.value}_{cmd_str}_velocities.csv")
    fall_log_path = _HERE / "logs" / f"{_RANDOM_SEED.value:03d}_{_ONNX_MODEL.value}_bl{_BACKLASH.value}_fall_times.csv"
    fallen = False
    # Clear previous log
    with open(velocity_log_path, "w") as f:
        while sim_time < _MAX_TIME.value:
            mujoco.mj_step(model, data)
            sim_time += 0.002

            # Check if fallen - same logic as in joystick.py
            gravity_vector = data.sensor("upvector").data
            if gravity_vector[2] < 0.0:
                #logging.info("Robot has fallen! Gravity z-component: %.3f", gravity_vector[2])
                f.write(f"{cmd_str},{sim_time:.3f}\n")
                fallen = True
                break

            # Check for NaN values in position or velocity
            if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
                logging.info("NaN detected in simulation state!")
                break

            # measure real velocity:
            linvel = data.sensor("local_linvel").data
            angular_vel = data.sensor("gyro").data
            print(_DISABLE_VELOCITY_LOG.value, "velolog")
            if not _DISABLE_VELOCITY_LOG.value:
                f.write(",".join(map(str, linvel)) + "," + ",".join(map(str, angular_vel)) + "\n")

            if mj_viewer and (sim_time - last_sync_time) > (1.0 / 60.0):
                mj_viewer.sync()
                last_sync_time = sim_time
                if _REALTIME.value:
                    real_elapsed = time.time() - real_start_time
                    if sim_time > real_elapsed:
                        time.sleep(sim_time - real_elapsed)
        if not fallen:
            f.write(f"{cmd_str},{-1.0:.3f}\n")


def main(argv):
    del argv  # unused
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        wolfgang_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=get_assets(),
    )

    np.random.seed(_RANDOM_SEED.value)
    # Set backlash joint ranges
    backlash_value = _BACKLASH.value
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and joint_name.endswith("_backlash"):
            joint_id = model.joint(joint_name).id
            if not _RANDOM_BACKLASH.value:
                model.jnt_range[joint_id] = [-backlash_value, backlash_value]
            else:
                model.jnt_range[joint_id] = (np.random.uniform(
                    -backlash_value, 0), np.random.uniform(0, backlash_value))

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.002
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy = OnnxController(
        policy_path=(_ONNX_DIR / _ONNX_MODEL.value).as_posix(),
        default_angles=np.array(model.keyframe("home").qpos[7::2]),
        ctrl_dt=ctrl_dt,
        n_substeps=n_substeps,
        action_scale=0.5,
    )

    mujoco.set_mjcb_control(policy.get_control)

    commands = [
        (0.0, 0.0, 0.0),  # stand still
        (0.2, 0.0, 0.0),  # forward
        (0.4, 0.0, 0.0),  # forward
        (-0.2, 0.0, 0.0),  # backward
        (-0.4, 0.0, 0.0),  # backward
        (0.0, 0.2, 0.0),  # left
        (0.0, 0.4, 0.0),  # left
        (0.0, -0.2, 0.0),  # right
        (0.0, -0.4, 0.0),  # right
        (0.0, 0.0, 0.5),  # turn left
        (0.0, 0.0, 1.0),  # turn left
        (0.0, 0.0, 1.5),  # turn left
        (0.0, 0.0, -0.5),  # turn right
        (0.0, 0.0, -1.0),  # turn right
        (0.0, 0.0, -1.5),  # turn right
    ]
    if _VISUALIZE.value:
        with viewer.launch_passive(model, data) as mj_viewer:
            for command in commands:
                run_experiment(command, policy, model, data, mj_viewer)
            mj_viewer.sync()
            mj_viewer.close()
    else:
        for command in commands:
            run_experiment(command, policy, model, data, None)

if __name__ == "__main__":
    app.run(main)
