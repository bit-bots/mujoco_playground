import os
os.environ['MUJOCO_GL'] = 'egl'  # Set the environment variable for EGL rendering

import numpy as np
import matplotlib.pyplot as plt
import datetime

from datetime import datetime
import functools
import os
import mediapy as media
from etils import epath
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import numpy as np
from absl import app
from absl import flags
from absl import logging
from mujoco_playground._src.gait import draw_joystick_command

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground import wrapper_torch
from mujoco_playground.experimental.utils.plotting import TrainingPlotter
import mujoco
import torch

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
_CONFIG_OVERRIDES = flags.DEFINE_string(
    "config_overrides",
    None,
    "Configuration overrides as comma-separated key=value pairs (e.g., 'key1=value1,key2=value2').",
)
_RL_CONFIG_OVERRIDES = flags.DEFINE_string(
    "rl_config_overrides",
    None,
    "RL config overrides as comma-separated key=value pairs.",
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")



def parse_config_overrides(config_str):
    """Parse comma-separated key=value pairs into a dict."""
    if not config_str:
        return {}
    overrides = {}
    for pair in config_str.split(","):
        key, value = pair.split("=")
        key = key.strip()
        value = value.strip()
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


def get_rl_config(env_name: str) -> dict:
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


def save_onnx_from_rsl_rl(ckpt_path, runner, raw_env, run_name, env_name):
    """Export RSL-RL inference policy directly to ONNX format.
    
    This function extracts the actor network and normalization from RSL-RL
    and exports them as a single ONNX model.
    """
    try:
        import torch.onnx
        
        # Get the policy module
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
        
        # Create dummy input for ONNX export (concatenated actor observations)
        dummy_input = torch.zeros((1, num_actor_obs), dtype=torch.float32, device=runner.device)
        
        # Export to ONNX
        onnx_path = ckpt_path / f"{run_name}.onnx"
        print(f"Exporting RSL-RL policy to ONNX: {onnx_path}")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Actor observation size: {num_actor_obs}")
        
        # Move model to CPU for ONNX export (ONNX export works better on CPU)
        wrapper_model_cpu = wrapper_model.cpu()
        dummy_input_cpu = dummy_input.cpu()
        
        # Test forward pass before export
        with torch.no_grad():
            test_output = wrapper_model_cpu(dummy_input_cpu)
            print(f"  Output shape: {test_output.shape}")
        
        torch.onnx.export(
            wrapper_model_cpu,
            dummy_input_cpu,
            str(onnx_path),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=11,
            do_constant_folding=True,
            dynamic_axes={
                "obs": {0: "batch_size"},
                "actions": {0: "batch_size"}
            },
        )
        
        print(f"Successfully exported ONNX model to {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(argv):
    """Run training and evaluation for the specified environment using RSL-RL."""
    del argv  # unused
    
    if _RUN_NAME.value is None:
        raise ValueError("--run_name is required")
    
    device = _DEVICE.value
    device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

    
    # Parse config overrides
    config_overrides = parse_config_overrides(_CONFIG_OVERRIDES.value)
    rl_config_overrides = parse_config_overrides(_RL_CONFIG_OVERRIDES.value)
    
    # Generate checkpoint path
    ckpt_path = epath.Path(__file__).parent / "checkpoints" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_RUN_NAME.value}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    
    env_name = _ENV_NAME.value
    env_cfg = registry.get_default_config(env_name)
    
    # Get RL config
    train_cfg = get_rl_config(env_name)
    train_cfg.update(rl_config_overrides)
    
    num_envs = train_cfg.num_envs
    
    # Setup logging directory (RSL-RL uses logs/ directory)
    logdir = os.path.abspath(os.path.join("logs", f"{_RUN_NAME.value}"))
    os.makedirs(logdir, exist_ok=True)
    
    # Initialize Weights & Biases if required
    if _USE_WANDB.value:
        import wandb
        # RSL-RL logs to TensorBoard, so we patch wandb to capture those logs
        wandb.tensorboard.patch(root_logdir=logdir)
        run = wandb.init(
            project="mujoco_playground",
            entity="bitbots",
            name=_RUN_NAME.value,
            config={
                "env": env_name,
                "gpu": _DEVICE.value,
                "run_name": _RUN_NAME.value,
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
        _SEED.value,
        env_cfg.episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )
    
    # Build RSL-RL config
    obs_size = raw_env.observation_size
    if isinstance(obs_size, dict):
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
    else:
        train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}
    
    # Overwrite default config with flags
    train_cfg.seed = _SEED.value
    train_cfg.run_name = _RUN_NAME.value
    
    train_cfg_dict = train_cfg.to_dict()
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(brax_env, train_cfg_dict, logdir, device=device)
    
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
    
    try:
        print("Rendering Video")
        
        # Get inference policy
        policy = runner.get_inference_policy(device=device)
        
        eval_env = registry.load(env_name, config_overrides=config_overrides)
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        
        rng = jax.random.PRNGKey(_SEED.value)
        
        rollout = []
        modify_scene_fns = []
        
        commands = jp.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [-0.5, 0.0, 0.0],
                            [0.0, 0.4, 0.0],
                            [0.0, -0.4, 0.0],
                            [0.0, 0.0, 2.0],
                            [0.0, 0.0, -2.0],])
        phase_dt = 2 * jp.pi * eval_env.dt * 1.5
        phase = jp.array([0, jp.pi])
        
        for j in range(commands.shape[0]):
            print(f"episode {j}")
            state = jit_reset(rng)
            state.info["phase_dt"] = phase_dt
            state.info["phase"] = phase
            for i in range(env_cfg.episode_length):
                state.info["command"] = commands[j]
                # Get observation in torch format
                # Policy expects a dict with observation groups as keys
                if isinstance(state.obs, dict):
                    # Convert dict observations to torch
                    obs_torch_dict = {k: wrapper_torch._jax_to_torch(v) for k, v in state.obs.items()}
                else:
                    # Single observation - wrap in dict with "state" key
                    obs_torch = wrapper_torch._jax_to_torch(state.obs)
                    obs_torch_dict = {"state": obs_torch}
                
                with torch.no_grad():
                    actions = policy(obs_torch_dict)
                # Convert back to JAX and flatten
                ctrl = wrapper_torch._torch_to_jax(actions.squeeze(0) if actions.dim() > 1 else actions)
                state = jit_step(state, ctrl)
                if state.done:
                    break
                rollout.append(state)
                torso_body_name = "pelvis_link"
                if env_name.startswith("Wolfgang"):
                    torso_body_name = "torso"
                xyz = np.array(state.data.xpos[eval_env.mj_model.body(torso_body_name).id])
                xyz += np.array([0, 0.0, 0])
                x_axis = state.data.xmat[eval_env._torso_body_id, 0]
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=state.info["command"],
                        xyz=xyz,
                        theta=yaw,
                        scl=1.0,
                    )
                )
        
        render_every = 2
        fps = 1.0 / eval_env.dt / render_every
        print(f"fps: {fps}")
        traj = rollout[::render_every]
        mod_fns = modify_scene_fns[::render_every]
        
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        
        frames = eval_env.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640,
            height=480*2,
            modify_scene_fns=mod_fns,
        )
        media.write_video(ckpt_path / f"{_RUN_NAME.value}_eval.mp4", frames, fps=fps)
        if _USE_WANDB.value:
            wandb.log({"video": wandb.Video(str(ckpt_path / f"{_RUN_NAME.value}_eval.mp4"), fps=fps, format="mp4")})
    except Exception as e:
        print(f"Failed to render video: {e}")
        import traceback
        traceback.print_exc()
    # Export RSL-RL policy directly to ONNX
    print("Exporting RSL-RL policy to ONNX...")
    onnx_path = save_onnx_from_rsl_rl(ckpt_path, runner, raw_env, _RUN_NAME.value, env_name)
    
    if onnx_path and onnx_path.exists():
        if _USE_WANDB.value:
            wandb.log_artifact(str(onnx_path), name="onnx_model", type="model")
        print(f"ONNX model saved successfully: {onnx_path}")
    else:
        print("Failed to export ONNX model.")


if __name__ == "__main__":
    app.run(main)

