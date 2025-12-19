"""Evaluate an ONNX policy with different velocity commands."""

from absl import app
from absl import flags
from absl import logging
from etils import epath
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.locomotion.wolfgang import wolfgang_constants
from mujoco_playground._src.locomotion.wolfgang.base import get_assets

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"

_BACKLASH = flags.DEFINE_float(
    "backlash", 0.05, "Backlash value for backlash joints (in radians)"
)
_ONNX_MODEL = flags.DEFINE_string(
    "onnx_model", "wolfgang_policy.onnx", "Name of the ONNX model file"
)
_MAX_TIME = flags.DEFINE_float(
    "max_time", 30.0, "Maximum simulation time per test (in seconds)"
)
_VISUALIZE = flags.DEFINE_boolean(
    "visualize", True, "Whether to visualize the evaluation"
)


class OnnxController:
  """ONNX controller for the wolfgang humanoid with fixed command."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: float = 0.5,
      command: np.ndarray = None,
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

    # Fixed command: [linvel_x, linvel_y, angvel_yaw]
    self._command = (
        np.array(command, dtype=np.float32) if command is not None
        else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

  def set_command(self, command: np.ndarray):
    """Set the velocity command."""
    self._command = np.array(command, dtype=np.float32)

  def get_obs(self, model, data) -> np.ndarray:
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])

    joint_angles = data.qpos[7::2] - self._default_angles
    joint_velocities = data.qvel[6::2]
    phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
    obs = np.hstack([
        gyro,
        gravity,
        self._command,  # Use fixed command instead of joystick
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

  def reset(self):
    """Reset controller state."""
    self._counter = 0
    self._last_action = np.zeros_like(self._default_angles, dtype=np.float32)
    self._phase = np.array([0.0, np.pi])


def check_fall(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
  """Check if the robot has fallen."""
  # Get gravity vector in local frame
  imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
  gravity_local = imu_xmat.T @ np.array([0, 0, -1])
  
  # Robot falls if gravity vector z-component is negative (pointing up)
  fall_termination = gravity_local[2] < 0.0
  
  # Also check for NaN values
  has_nan = np.isnan(data.qpos).any() or np.isnan(data.qvel).any()
  
  return bool(fall_termination or has_nan)


class EvaluationState:
  """State for evaluation with visualization."""
  
  def __init__(self, model, data, controller, test_commands, max_time, sim_dt):
    self.model = model
    self.data = data
    self.controller = controller
    self.test_commands = test_commands
    self.max_time = max_time
    self.sim_dt = sim_dt
    
    self.current_test_idx = 0
    self.results = []
    self.step_count = 0
    self.actual_velocities = []
    self.test_started = False
    self.test_finished = False


def evaluate_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: OnnxController,
    command: np.ndarray,
    max_time: float,
    sim_dt: float,
) -> dict:
  """Evaluate a single velocity command."""
  controller.set_command(command)
  controller.reset()
  
  # Reset simulation
  mujoco.mj_resetDataKeyframe(model, data, 1)
  
  # Track metrics
  actual_velocities = []
  times = []
  time_to_fall = None
  
  max_steps = int(max_time / sim_dt)
  
  for step in range(max_steps):
    # Apply control
    controller.get_control(model, data)
    
    # Step simulation
    mujoco.mj_step(model, data)
    
    # Record actual velocity
    actual_linvel = data.sensor("local_linvel").data.copy()
    actual_velocities.append(actual_linvel)
    times.append(step * sim_dt)
    
    # Check for fall
    if check_fall(model, data):
      time_to_fall = step * sim_dt
      break
  
  # Compute average velocity (excluding initial transient)
  if len(actual_velocities) > 100:  # Skip first ~2 seconds
    avg_velocities = np.mean(actual_velocities[100:], axis=0)
  else:
    avg_velocities = np.mean(actual_velocities, axis=0) if actual_velocities else np.array([0.0, 0.0, 0.0])
  
  return {
      "command": command.copy(),
      "time_to_fall": time_to_fall if time_to_fall is not None else max_time,
      "avg_actual_velocity": avg_velocities,
      "final_time": times[-1] if times else 0.0,
      "fell": time_to_fall is not None,
  }


def make_control_callback(eval_state: EvaluationState):
  """Create a control callback for visualization."""
  def control_callback(model, data):
    if eval_state.test_finished:
      # Move to next test
      eval_state.current_test_idx += 1
      eval_state.test_finished = False
      eval_state.test_started = False
      eval_state.step_count = 0
      eval_state.actual_velocities = []
      
      if eval_state.current_test_idx >= len(eval_state.test_commands):
        # All tests done - pause simulation
        return
    
    if not eval_state.test_started:
      # Start new test
      cmd = eval_state.test_commands[eval_state.current_test_idx]
      eval_state.controller.set_command(np.array(cmd))
      eval_state.controller.reset()
      mujoco.mj_resetDataKeyframe(eval_state.model, eval_state.data, 1)
      eval_state.test_started = True
      logging.info(f"Test {eval_state.current_test_idx+1}/{len(eval_state.test_commands)}: Command {cmd}")
    
    # Apply control
    eval_state.controller.get_control(model, data)
    
    # Record metrics
    actual_linvel = data.sensor("local_linvel").data.copy()
    eval_state.actual_velocities.append(actual_linvel)
    eval_state.step_count += 1
    
    # Check for fall or timeout
    current_time = eval_state.step_count * eval_state.sim_dt
    max_steps = int(eval_state.max_time / eval_state.sim_dt)
    
    if check_fall(model, data) or eval_state.step_count >= max_steps:
      # Finish current test
      cmd = eval_state.test_commands[eval_state.current_test_idx]
      
      # Compute average velocity
      if len(eval_state.actual_velocities) > 100:
        avg_velocities = np.mean(eval_state.actual_velocities[100:], axis=0)
      else:
        avg_velocities = (
            np.mean(eval_state.actual_velocities, axis=0)
            if eval_state.actual_velocities
            else np.array([0.0, 0.0, 0.0])
        )
      
      time_to_fall = current_time if check_fall(model, data) else None
      
      result = {
          "command": cmd.copy(),
          "time_to_fall": time_to_fall if time_to_fall is not None else eval_state.max_time,
          "avg_actual_velocity": avg_velocities,
          "final_time": current_time,
          "fell": time_to_fall is not None,
      }
      eval_state.results.append(result)
      
      # Print result
      cmd_str = f"[{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]"
      avg_vel_str = f"[{avg_velocities[0]:.3f}, {avg_velocities[1]:.3f}, {avg_velocities[2]:.3f}]"
      status = f"Fell at {time_to_fall:.2f}s" if result['fell'] else f"Survived {current_time:.2f}s"
      
      logging.info(f"  Command: {cmd_str}")
      logging.info(f"  Avg actual velocity: {avg_vel_str}")
      logging.info(f"  Status: {status}")
      logging.info("")
      
      eval_state.test_finished = True
  
  return control_callback


def main(argv):
  del argv  # unused

  # Load model
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
  mujoco.mj_resetDataKeyframe(model, data, 1)

  ctrl_dt = 0.02
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  # Load controller
  policy_path = (_ONNX_DIR / _ONNX_MODEL.value).as_posix()
  if not epath.Path(policy_path).exists():
    logging.error(f"ONNX model not found: {policy_path}")
    return

  controller = OnnxController(
      policy_path=policy_path,
      default_angles=np.array(model.keyframe("home").qpos[7::2]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.5,
  )

  # Define test commands: [linvel_x, linvel_y, angvel_yaw]
  # User can modify this list later
  test_commands = [
      [0.0, 0.0, 0.0],  # Stand still
      [0.1, 0.0, 0.0],  # Forward 0.1 m/s
      [0.2, 0.0, 0.0],  # Forward 0.2 m/s
      [0.3, 0.0, 0.0],  # Forward 0.3 m/s
      [0.4, 0.0, 0.0],  # Forward 0.4 m/s
      [0.5, 0.0, 0.0],  # Forward 0.5 m/s
      [0.0, 0.1, 0.0],  # Lateral 0.1 m/s
      [0.0, 0.0, 0.5],  # Rotate 0.5 rad/s
      [0.0, 0.0, 1.0],  # Rotate 1.0 rad/s
  ]

  # Run evaluations
  logging.info(f"Evaluating {len(test_commands)} commands...")
  logging.info(f"Using ONNX model: {_ONNX_MODEL.value}")
  logging.info(f"Backlash value: {_BACKLASH.value}")
  logging.info(f"Max time per test: {_MAX_TIME.value}s")
  logging.info("-" * 80)

  if _VISUALIZE.value:
    # Use viewer for visualization
    eval_state = EvaluationState(
        model, data, controller, test_commands, _MAX_TIME.value, sim_dt
    )
    control_callback = make_control_callback(eval_state)
    
    def load_callback(viewer_model=None, viewer_data=None):
      mujoco.set_mjcb_control(None)
      mujoco.set_mjcb_control(control_callback)
      return model, data
    
    logging.info("Starting visualization...")
    logging.info("Close the viewer window when evaluation is complete.")
    mujoco.viewer.launch(loader=load_callback)
    results = eval_state.results
  else:
    # Run without visualization
    results = []
    for i, cmd in enumerate(test_commands):
      logging.info(f"Test {i+1}/{len(test_commands)}: Command {cmd}")
      result = evaluate_command(
          model, data, controller, np.array(cmd), _MAX_TIME.value, sim_dt
      )
      results.append(result)
      
      # Print result
      cmd_str = f"[{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]"
      avg_vel_str = f"[{result['avg_actual_velocity'][0]:.3f}, {result['avg_actual_velocity'][1]:.3f}, {result['avg_actual_velocity'][2]:.3f}]"
      status = f"Fell at {result['time_to_fall']:.2f}s" if result['fell'] else f"Survived {result['final_time']:.2f}s"
      
      logging.info(f"  Command: {cmd_str}")
      logging.info(f"  Avg actual velocity: {avg_vel_str}")
      logging.info(f"  Status: {status}")
      logging.info("")

  # Print summary
  logging.info("=" * 80)
  logging.info("SUMMARY")
  logging.info("=" * 80)
  logging.info(f"{'Command':<30} {'Avg Vel (x,y,yaw)':<30} {'Time':<15} {'Status':<10}")
  logging.info("-" * 80)
  
  for result in results:
    cmd = result["command"]
    cmd_str = f"[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}]"
    vel_str = f"[{result['avg_actual_velocity'][0]:.3f},{result['avg_actual_velocity'][1]:.3f},{result['avg_actual_velocity'][2]:.3f}]"
    time_str = f"{result['time_to_fall']:.2f}s" if result['fell'] else f"{result['final_time']:.2f}s"
    status_str = "FELL" if result['fell'] else "OK"
    
    logging.info(f"{cmd_str:<30} {vel_str:<30} {time_str:<15} {status_str:<10}")


if __name__ == "__main__":
  app.run(main)

