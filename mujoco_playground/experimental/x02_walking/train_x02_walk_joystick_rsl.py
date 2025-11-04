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
import pickle
from tqdm import tqdm
from mujoco_playground.experimental.x02_walking.convert_to_onnx import conv_to_onnx
from mujoco_playground._src.gait import draw_joystick_command

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import mujoco_playground
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


def save_params_brax_format(ckpt_path, runner, raw_env, step=-1):
    """Attempt to save parameters in Brax-compatible format for ONNX conversion.
    
    WARNING: This is a workaround. RSL-RL models are in PyTorch format,
    while ONNX conversion expects Brax format. This may not work correctly.
    """
    # Try to extract normalization stats from the runner
    # This is a simplified approach - may need adjustment based on RSL-RL implementation
    try:
        # Get the actor network to extract normalization stats
        if hasattr(runner, "alg") and hasattr(runner.alg, "actor"):
            # RSL-RL uses empirical normalization, stats are stored in the runner
            # This is a placeholder - actual implementation depends on RSL-RL internals
            normalizer_params = None
            policy_params = None
            value_params = None
            
            filename = ckpt_path / f"params_{step:012}.pkl" if step >= 0 else ckpt_path / "params.pkl"
            with open(filename, "wb") as f:
                data = {
                    "normalizer_params": normalizer_params,
                    "policy_params": policy_params,
                    "value_params": value_params,
                }
                pickle.dump(data, f)
            print(f"WARNING: Saved placeholder params to {filename}")
            print("ONNX conversion may not work with RSL-RL models directly.")
    except Exception as e:
        print(f"Could not save Brax-format params: {e}")


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
            # Handle both dict and array observations
            if isinstance(state.obs, dict):
                obs_data = state.obs.get("state", state.obs)
            else:
                obs_data = state.obs
            obs_torch = wrapper_torch._jax_to_torch(obs_data)
            with torch.no_grad():
                actions = policy(obs_torch.unsqueeze(0))
            # Convert back to JAX and flatten
            ctrl = wrapper_torch._torch_to_jax(actions.squeeze(0))
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
    
    # Note: RSL-RL models are in PyTorch format, not Brax format
    # The ONNX conversion function expects Brax format, so it won't work directly
    # You would need to convert the PyTorch model to ONNX separately
    print("Note: ONNX conversion expects Brax format parameters.")
    print("RSL-RL models are in PyTorch format. To convert to ONNX, you would need")
    print("to extract the PyTorch model from the runner and convert it separately.")
    
    # Try to save in Brax format for ONNX conversion (may not work perfectly)
    try:
        save_params_brax_format(ckpt_path, runner, raw_env)
        conv_to_onnx(ckpt_path / "params.pkl", f"{_RUN_NAME.value}.onnx", env_name)
        if _USE_WANDB.value:
            wandb.log_artifact(str(f"{_RUN_NAME.value}.onnx"), name="onnx_model", type="model")
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        print("This is expected - RSL-RL models are in PyTorch format, not Brax format.")


if __name__ == "__main__":
    app.run(main)

