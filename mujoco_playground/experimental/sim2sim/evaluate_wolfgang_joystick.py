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
_ONNX_MODEL = flags.DEFINE_string(
    "onnx_model", "wolfgang_policy_bl005.onnx", "Name of the ONNX model file"
)
_MAX_TIME = flags.DEFINE_float(
    "max_time", 3.0, "Maximum simulation time per test (in seconds)"
)
_VISUALIZE = flags.DEFINE_boolean(
    "visualize", True, "Whether to visualize the evaluation"
)
_REALTIME = flags.DEFINE_boolean(
    "realtime", False, "Whether to run the simulation in real-time"
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
  logging.info("Test command: %s", command)
  policy.set_command(command)
  policy.reset()
  mujoco.mj_resetDataKeyframe(model, data, 0)
  sim_time = 0.0
  last_sync_time = -1.0
  real_start_time = time.time()
  cmd_str = "_".join([f"{c:.2f}" for c in command])
  velocity_log_path = _HERE / "logs" / f"{_ONNX_MODEL.value}_bl{_BACKLASH.value}_{cmd_str}_velocities.csv"
  # Clear previous log
  with open(velocity_log_path, "w") as f:
    f.write("")
  
  while sim_time < _MAX_TIME.value:
    mujoco.mj_step(model, data)
    sim_time += 0.002
    
    # Check if fallen - same logic as in joystick.py
    gravity_vector = data.sensor("upvector").data
    if gravity_vector[2] < 0.0:
      logging.info("Robot has fallen! Gravity z-component: %.3f", gravity_vector[2])
      break
    
    # Check for NaN values in position or velocity
    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
      logging.info("NaN detected in simulation state!")
      break

    # measure real velocity:
    linvel = data.sensor("local_linvel").data
    angular_vel = data.sensor("gyro").data

    with open(velocity_log_path, "a") as f:
        f.write(",".join(map(str, linvel)) + "," + ",".join(map(str, angular_vel)) + "\n")

    if mj_viewer and (sim_time - last_sync_time) > (1.0 / 60.0):
      mj_viewer.sync()
      last_sync_time = sim_time
      if _REALTIME.value:
        real_elapsed = time.time() - real_start_time
        if sim_time > real_elapsed:
          time.sleep(sim_time - real_elapsed)
  logging.info("Fini testing command: %s", command)

def main(argv):
  del argv  # unused
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      wolfgang_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )

  # Set backlash joint ranges
  backlash_value = _BACKLASH.value
  for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if joint_name and joint_name.endswith("_backlash"):
      joint_id = model.joint(joint_name).id
      model.jnt_range[joint_id] = [-backlash_value, backlash_value]

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
      (0.5, 0.0, 0.0),  # forward
      (0.0, 0.5, 0.0),  # left
      (0.0, -0.5, 0.0),  # right
      (0.0, 0.0, 0.5),  # turn left
      (0.0, 0.0, -0.5),  # turn right
  ]
  
  with viewer.launch_passive(model, data) as mj_viewer:
    for command in commands:
      run_experiment(command, policy, model, data, mj_viewer)
    mj_viewer.close()

if __name__ == "__main__":
  app.run(main)
