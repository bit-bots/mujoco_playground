# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
import numpy as np

from mujoco_playground._src.locomotion.x02 import x02_constants
from mujoco_playground._src.locomotion.x02.base import get_assets
#from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad
from mujoco_playground.experimental.sim2sim.keyboard_gamepad import KeyboardGamepad as Gamepad
from argparse import ArgumentParser

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"

parser = ArgumentParser()
parser.add_argument(
    "onnx_path",
    type=str,
    default=(_ONNX_DIR / "history10").as_posix(),
    help="Path to the ONNX policy.",
)
parser.add_argument(
    "--history_len",
    type=int,
    default=1,
    help="History length.",
)
args = parser.parse_args()

class OnnxController:
  """ONNX controller for the DroidUp X02 humanoid."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.0,
      vel_scale_y: float = 1.0,
      vel_scale_rot: float = 1.0,
      history_len: int = 1,
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
    self._gait_freq = 1.25
    self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt

    self._joystick = Gamepad(
        vel_scale_x=vel_scale_x,
        vel_scale_y=vel_scale_y,
        vel_scale_rot=vel_scale_rot,
    )
    self._history_len = history_len
    self._qvel_history = np.zeros((self._history_len, 10))
    self._qpos_error_history = np.zeros((self._history_len, 10))

  def get_obs(self, model, data) -> np.ndarray:
    # Update history buffers (roll and insert new values at the beginning)
    # Roll the history arrays to shift old values
    self._qvel_history = np.roll(self._qvel_history, 10, axis=0)
    self._qpos_error_history = np.roll(self._qpos_error_history, 10, axis=0)
    # Insert new values at the beginning
    self._qvel_history[:10] = data.qvel[6:]
    self._qpos_error_history[:10] = data.qpos[7:] - (self._last_action * self._action_scale + self._default_angles)
    
    # Flatten history for observation
    qvel_history = self._qvel_history.flatten()
    qpos_error_history = self._qpos_error_history.flatten()
    
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
    command = self._joystick.get_command()
    obs = np.hstack([
        #linvel,
        qvel_history, # 10*history_len
        qpos_error_history, # 10*history_len
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


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      x02_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  ctrl_dt = 0.02
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / (args.onnx_path + ".onnx")).as_posix(),
      default_angles=np.array(model.keyframe("home").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.5,
      vel_scale_x=1.0,
      vel_scale_y=1.0,
      vel_scale_rot=1.0,
      history_len=args.history_len,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
