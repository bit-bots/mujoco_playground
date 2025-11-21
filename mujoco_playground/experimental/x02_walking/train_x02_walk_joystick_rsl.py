import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Set the environment variable for EGL rendering

import numpy as np
import datetime

from datetime import datetime
import functools
import os
import mediapy as media
from etils import epath
import jax
from jax import numpy as jp
import numpy as np
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_dict

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground import wrapper_torch
from mujoco_playground._src.gait import draw_joystick_command
import mujoco
import torch

from rsl_rl.runners import OnPolicyRunner

# Suppress logs if you want
logging.set_verbosity(logging.WARNING)

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

# Define flags
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "X02JoystickFlatTerrain",
    (
        "Name of the environment. One of: "
        f"{', '.join(registry.ALL_ENVS)}"
    ),
)
_RUN_NAME = flags.DEFINE_string(
    "run_name", None, "Name of the run for saving parameters (required)."
)
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training.")
_USE_WANDB = flags.DEFINE_boolean(
    "wandb",
    False,
    "Enable Weights & Biases logging.",
)
_CONFIG_OVERRIDES = flags.DEFINE_spaceseplist(
    "config_overrides",
    [],
    "Configuration overrides as key=value pairs (e.g., 'key1=value1 key2=value2').",
)
_RL_CONFIG_OVERRIDES = flags.DEFINE_spaceseplist(
    "rl_config_overrides",
    [],
    "RL config overrides as key=value pairs (e.g., 'key1=value1 key2=value2').",
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_RENDER_VIDEO = flags.DEFINE_boolean("render_video", False, "Render video.")



def parse_config_overrides(config):
    """Parse comma-separated key=value pairs into a dict."""
    if not config:
        return {}
    overrides = {}
    print(f"Parsing config overrides: {config}")
    for pair in config:
        key, value = pair.split("=")
        key = key.strip()
        value = value.strip()
        if value.startswith("["):
            value = eval(value)
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
        overrides[key] = value
    return overrides


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    """Get RL config for the environment."""
    if env_name in registry.manipulation._envs:
        from mujoco_playground.config import manipulation_params
        return manipulation_params.rsl_rl_config(env_name)
    elif env_name in registry.locomotion._envs:
        return locomotion_params.rsl_rl_config(env_name)
    else:
        raise ValueError(f"No RL config for {env_name}")


def save_params_rsl_rl(ckpt_path, runner, step=-1):
    """Save RSL-RL model parameters.
    
    Note: RSL-RL uses PyTorch models, so the format differs from Brax.
    This saves the runner's model state, which can be loaded later.
    """
    # RSL-RL saves checkpoints automatically, but we can also save here
    # The runner.save() method is typically called during training
    filename = ckpt_path / f"model_{step:012}.pt" if step >= 0 else ckpt_path / "model.pt"
    runner.save(str(filename))
    print(f"Saved RSL-RL model to {filename}")


def save_onnx_from_rsl_rl(ckpt_path, runner : OnPolicyRunner, raw_env, run_name, env_name, step=None):
    """Export RSL-RL inference policy directly to ONNX format.
    
    This function extracts the actor network and normalization from RSL-RL
    and exports them as a single ONNX model.
    
    Args:
        ckpt_path: Path to checkpoint directory
        runner: RSL-RL runner instance
        raw_env: Environment instance
        run_name: Name of the run
        env_name: Name of the environment
        step: Optional step number for intermediate checkpoints (if None, uses final name)
    """
    try:
        import torch.onnx
        
        # Get the policy module (DO NOT move to CPU - it will affect training!)
        policy = runner.alg.policy
        policy.eval()  # Set to evaluation mode
        
        # Get observation size
        obs_size = raw_env.observation_size
        if isinstance(obs_size, dict):
            # Get actor observation size
            # obs_size values are tuples (shapes), extract the last dimension
            obs_groups = policy.obs_groups["policy"]
            num_actor_obs = 0
            for obs_group in obs_groups:
                obs_shape = obs_size[obs_group]
                if isinstance(obs_shape, tuple):
                    # Extract the last dimension (feature dimension)
                    num_actor_obs += obs_shape[-1]
                elif isinstance(obs_shape, int):
                    num_actor_obs += obs_shape
                else:
                    # Try to get the size if it's a JAX array shape
                    num_actor_obs += int(obs_shape[-1])
        else:
            # obs_size is an int or a shape tuple
            if isinstance(obs_size, tuple):
                num_actor_obs = obs_size[-1]
            else:
                num_actor_obs = int(obs_size)
        
        # Create a wrapper model that includes normalization and actor
        # ONNX doesn't support dict inputs, so we use a single tensor input
        class PolicyWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
            
            def forward(self, obs):
                # Input is already concatenated actor observations as a single tensor
                # Apply normalization
                normalized_obs = self.policy.actor_obs_normalizer(obs)
                
                # Forward through actor
                if self.policy.state_dependent_std:
                    output = self.policy.actor(normalized_obs)
                    # Return mean action (first half of output)
                    return output[..., 0, :]
                else:
                    return self.policy.actor(normalized_obs)
        
        wrapper_model = PolicyWrapper(policy)
        wrapper_model.eval()
        
        # Export to ONNX with step number if provided
        if step is not None:
            onnx_path = ckpt_path / f"{run_name}_step_{step:012}.onnx"
        else:
            onnx_path = ckpt_path / f"{run_name}.onnx"
        print(f"Exporting RSL-RL policy to ONNX: {onnx_path}")
        print(f"  Actor observation size: {num_actor_obs}")
        
        # Create dummy input for ONNX export on the same device as the model
        dummy_input = torch.zeros((1, num_actor_obs), dtype=torch.float32, device=runner.device)
        
        # Test forward pass before export
        with torch.no_grad():
            test_output = wrapper_model(dummy_input)
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {test_output.shape}")
        
        # Export directly on GPU - ONNX export supports GPU
        # This avoids any device state changes that could affect training
        torch.onnx.export(
            wrapper_model,
            dummy_input,
            str(onnx_path),
            input_names=["obs"],
            output_names=["continuous_actions"],
            opset_version=11,
            do_constant_folding=True,
            dynamic_axes={
                "obs": {0: "batch_size"},
                "continuous_actions": {0: "batch_size"}
            },
        )
        
        print(f"Successfully exported ONNX model to {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(argv,
         env_name: str,
         run_name: str,
         device: str,
         config_overrides: list[str] | dict,
         rl_config_overrides: list[str] | dict,
         use_wandb: bool,
         seed: int,
         render_video: bool,
         run_eval: bool,
         export_onnx: bool,
         config_is_string: bool,
         ):
    """Run training and evaluation for the specified environment using RSL-RL."""
    del argv  # unused
    
    device_rank = 0 #int(device.split(":")[-1]) if "cuda" in device else 0
    
    if isinstance(config_overrides, list):
        config_overrides = parse_config_overrides(config_overrides)
    if isinstance(rl_config_overrides, list):
        rl_config_overrides = parse_config_overrides(rl_config_overrides)
    
    # Generate checkpoint path
    ckpt_path = epath.Path(__file__).parent / "checkpoints" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    
    env_cfg = registry.get_default_config(env_name)
    
    # Get RL config
    train_cfg = get_rl_config(env_name)
    train_cfg.update_from_flattened_dict(rl_config_overrides)
    
    num_envs = train_cfg.num_envs
    
    # Use checkpoint directory as log directory
    logdir = str(ckpt_path)
    
    # Initialize Weights & Biases if required
    if use_wandb:
        import wandb
        # RSL-RL logs to TensorBoard, so we patch wandb to capture those logs
        wandb.tensorboard.patch(root_logdir=logdir)
        run = wandb.init(
            project="mujoco_playground",
            entity="bitbots",
            name=run_name,
            config={
                "env": env_name,
                "gpu": device,
                "run_name": run_name,
            } | dict(train_cfg.to_dict()) | dict(env_cfg.to_dict()) | config_overrides,
        )

    times = [datetime.now()]
    
    # Domain randomization
    randomizer = registry.get_domain_randomizer(env_name)
    
    # We'll store environment states during rendering
    render_trajectory = []
    
    # Callback to gather states for rendering
    def render_callback(_, state):
        render_trajectory.append(state)
    
    # Create the environment
    raw_env = registry.load(env_name, config_overrides=config_overrides)
    brax_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_envs,
        seed,
        env_cfg.episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )

    print(f"train_cfg: {train_cfg.to_dict()}")
    print(f"env_cfg: {raw_env._config.to_dict()}")
    
    # Build RSL-RL config
    obs_size = raw_env.observation_size
    if isinstance(obs_size, dict):
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
    else:
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}
    
    # Overwrite default config with flags
    train_cfg.seed = seed
    train_cfg.run_name = run_name
    
    train_cfg_dict = train_cfg.to_dict()
    runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)
    
    # Wrap the runner's save method to also export ONNX for intermediate checkpoints
    original_save = runner.save
    def save_with_onnx(path):
        """Wrapper around runner.save that also exports ONNX."""
        # Call original save method
        original_save(path)
        
        # Extract step number from path if available
        # RSL-RL saves as model_XXXXXX.pt where XXXXXX is the step number
        step = None
        if path:
            import re
            match = re.search(r'model_(\d+)\.pt', path)
            if match:
                step = int(match.group(1))
        
        # Export ONNX for this checkpoint
        if export_onnx:
            try:
                save_onnx_from_rsl_rl(ckpt_path, runner, raw_env, run_name, env_name, step=step)
            except Exception as e:
                print(f"Warning: Failed to export ONNX for checkpoint {path}: {e}")
    
    # Replace the save method
    runner.save = save_with_onnx
    
    # Perform training
    # Note: RSL-RL logs to TensorBoard automatically, and wandb patches to capture those logs
    print("Starting training...")
    runner.learn(
        num_learning_iterations=train_cfg.max_iterations,
        init_at_random_ep_len=False,
    )
    print("Done training.")
    
    times.append(datetime.now())
    if len(times) > 1:
        print(f"time to train: {times[-1] - times[0]}")
    
    # Save final model
    save_params_rsl_rl(ckpt_path, runner)
    
    if run_eval:
        policy = runner.get_inference_policy(device=device)
        eval_env = registry.load(env_name, config_overrides=config_overrides)
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)


        commands = np.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [-0.5, 0.0, 0.0],
                            [0.0, 0.4, 0.0],
                            [0.0, -0.4, 0.0],
                            [0.0, 0.0, 2.0],
                            [0.0, 0.0, -2.0],])
        sum_reward = 0.0
        sum_episode_length = 0
        for command in commands:
            rng = jax.random.PRNGKey(1)
            state = jit_reset(rng)
            # We’ll assume your environment’s observation is in state.obs["state"].
            obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])
            for _ in range(env_cfg.episode_length):
                with torch.no_grad():
                    actions = policy({"state": obs_torch})
                    # Step environment
                    state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
                    sum_episode_length += 1
                    sum_reward += state.reward
                    obs_torch = wrapper_torch._jax_to_torch(state.obs["state"])
                    if state.done:
                        break
        mean_reward = sum_reward / len(commands)
        mean_episode_length = sum_episode_length / len(commands)

    if render_video:
        print("Rendering Video...")
        # Get inference policy
        policy = runner.get_inference_policy(device=device)
        
        eval_env = registry.load(env_name, config_overrides=config_overrides)
        
        # Get mujoco model for rendering
        mj_model = eval_env.mj_model
        mj_data = mujoco.MjData(mj_model)
        
        # Get history length from config (default to 1)
        history_len = env_cfg.get("history_len", 1)
        
        # Initialize history buffers (similar to play_x02_joystick.py)
        qvel_history = np.zeros((history_len, 10))
        qpos_error_history = np.zeros((history_len, 10))
        default_angles = np.array(mj_model.keyframe("home").qpos[7:])
        last_action = np.zeros(10, dtype=np.float32)
        
        # Phase tracking
        phase = np.array([0.0, np.pi])
        gait_freq = 1.25
        phase_dt = 2 * np.pi * gait_freq * eval_env.dt
        
        # Reset mujoco data
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
        
        rollout_data = []
        modify_scene_fns = []
        
        commands = np.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [-0.5, 0.0, 0.0],
                            [0.0, 0.4, 0.0],
                            [0.0, -0.4, 0.0],
                            [0.0, 0.0, 2.0],
                            [0.0, 0.0, -2.0],])
        
        for j in range(commands.shape[0]):
            print(f"episode {j}")
            # Reset mujoco data
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 1)
            
            # Reset history buffers
            qvel_history.fill(0)
            qpos_error_history.fill(0)
            last_action.fill(0)
            phase = np.array([0.0, np.pi])
            
            command = commands[j]
            
            for i in range(env_cfg.episode_length//10):
                # Update history buffers (roll and insert new values)
                qvel_history = np.roll(qvel_history, 10, axis=0)
                qpos_error_history = np.roll(qpos_error_history, 10, axis=0)
                
                # Insert new values at the beginning
                qvel_history[:10] = mj_data.qvel[6:]
                qpos_error_history[:10] = mj_data.qpos[7:] - default_angles
                
                # Build observation (similar to play_x02_joystick.py)
                qvel_history_flat = qvel_history.flatten()
                qpos_error_history_flat = qpos_error_history.flatten()
                
                gyro = mj_data.sensor("gyro").data
                imu_site_id = mj_model.site("imu").id
                imu_xmat = mj_data.site_xmat[imu_site_id].reshape(3, 3)
                gravity = imu_xmat.T @ np.array([0, 0, -1])
                joint_angles = mj_data.qpos[7:] - default_angles
                joint_velocities = mj_data.qvel[6:]
                phase_cos_sin = np.concatenate([np.cos(phase), np.sin(phase)])
                
                obs = np.hstack([
                    qvel_history_flat,
                    qpos_error_history_flat,
                    gyro,
                    gravity,
                    command,
                    joint_angles,
                    joint_velocities,
                    last_action,
                    phase_cos_sin,
                ]).astype(np.float32)
                
                # Get action from policy
                obs_torch = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    actions = policy({"state": obs_torch})
                actions_np = actions.squeeze(0).cpu().numpy()
                last_action = actions_np.copy()
                
                # Apply action
                mj_data.ctrl[:] = actions_np * 0.5 + default_angles
                
                # Step simulation
                n_substeps = int(round(eval_env.dt / eval_env.sim_dt))
                for _ in range(n_substeps):
                    mujoco.mj_step(mj_model, mj_data)
                
                # Update phase
                phase_tp1 = phase + phase_dt
                phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
                
                # Store data for rendering
                rollout_data.append({
                    'qpos': mj_data.qpos.copy(),
                    'qvel': mj_data.qvel.copy(),
                    'mocap_pos': mj_data.mocap_pos.copy() if mj_model.nmocap > 0 else None,
                    'mocap_quat': mj_data.mocap_quat.copy() if mj_model.nmocap > 0 else None,
                    'xfrc_applied': mj_data.xfrc_applied.copy(),
                })
                
                # Get torso position for joystick visualization
                torso_body_name = "pelvis_link"
                if env_name.startswith("Wolfgang"):
                    torso_body_name = "torso"
                torso_body_id = mj_model.body(torso_body_name).id
                xyz = np.array(mj_data.xpos[torso_body_id])
                # xmat is flattened (9 elements), reshape to 3x3 and get first row (x-axis)
                xmat = mj_data.xmat[torso_body_id].reshape(3, 3)
                x_axis = xmat[0]
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=command,
                        xyz=xyz,
                        theta=yaw,
                        scl=1.0,
                    )
                )
        
        # Render using mujoco directly (not mjx)
        render_every = 2
        fps = 1.0 / eval_env.dt / render_every
        print(f"fps: {fps}")
        
        traj_data = rollout_data[::render_every]
        mod_fns = modify_scene_fns[::render_every]
        
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        
        # Create renderer
        renderer = mujoco.Renderer(mj_model, height=480*2, width=640)
        camera = mj_model.camera("track").id if mj_model.ncam > 0 else -1
        
        frames = []
        for i, data_dict in enumerate(traj_data):
            # Set mujoco data
            mj_data.qpos[:] = data_dict['qpos']
            mj_data.qvel[:] = data_dict['qvel']
            if data_dict['mocap_pos'] is not None:
                mj_data.mocap_pos[:] = data_dict['mocap_pos']
            if data_dict['mocap_quat'] is not None:
                mj_data.mocap_quat[:] = data_dict['mocap_quat']
            mj_data.xfrc_applied[:] = data_dict['xfrc_applied']
            
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera, scene_option=scene_option)
            
            if i < len(mod_fns):
                mod_fns[i](renderer.scene)
            
            frames.append(renderer.render())
        
        renderer.close()
        
        media.write_video(ckpt_path / f"{run_name}_eval.mp4", frames, fps=fps)
        if use_wandb:
            wandb.log({"video": wandb.Video(str(ckpt_path / f"{run_name}_eval.mp4"), fps=fps, format="mp4")})
    if export_onnx:
        # Export RSL-RL policy directly to ONNX
        print("Exporting RSL-RL policy to ONNX...")
        onnx_path = save_onnx_from_rsl_rl(ckpt_path, runner, raw_env, run_name, env_name)
        
        if onnx_path and onnx_path.exists():
            if use_wandb:
                wandb.log_artifact(str(onnx_path), name="onnx_model", type="model")
            print(f"ONNX model saved successfully: {onnx_path}")
        else:
            print("Failed to export ONNX model.")
    if run_eval:
        return mean_reward, mean_episode_length


if __name__ == "__main__":
    def main_wrapper(argv):
        """Wrapper to parse flags and call main with parsed values."""
        return main(
            argv,
            env_name=_ENV_NAME.value,
            run_name=_RUN_NAME.value,
            device=_DEVICE.value,
            config_overrides=_CONFIG_OVERRIDES.value,
            rl_config_overrides=_RL_CONFIG_OVERRIDES.value,
            use_wandb=_USE_WANDB.value,
            seed=_SEED.value,
            render_video=_RENDER_VIDEO.value,
            run_eval=False,
            export_onnx=True,
            config_is_string=True,
        )
    
    app.run(main_wrapper)

