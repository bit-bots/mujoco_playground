import os
os.environ['MUJOCO_GL'] = 'egl'  # Set the environment variable for EGL rendering

import numpy as np
import matplotlib.pyplot as plt
import datetime

from datetime import datetime
import functools
import os
import mediapy as media
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
#from brax.training.agents.sac import networks as sac_networks
#from brax.training.agents.sac import train as sac
from etils import epath
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from mujoco_playground.experimental.x02_walking.convert_to_onnx import conv_to_onnx
from mujoco_playground._src.gait import draw_joystick_command

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_dict

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
import mujoco


def parse_kv(s):
    key, value = s.split('=')
    try:
      value = int(value)
    except ValueError:
      try:
        value = float(value)
      except ValueError:
        if value.lower() == 'true':
          value = True
        elif value.lower() == 'false':
          value = False
    return key, value

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
parser = ArgumentParser(description="Train a walking agent with joystick control.")
parser.add_argument('-n', '--run-name', type=str, required=True, help='Name of the run for saving parameters')
parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU device to use')
parser.add_argument('-e', '--env', type=str, default='X02JoystickFlatTerrain')
parser.add_argument('-w', '--wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('-c', '--config', nargs="+", type=parse_kv, help='Overwrites for default configuration of environment' )
parser.add_argument('-C', "--rl-config", nargs="+", type=parse_kv, help='Overwrites for default configuration of RL algorithm' )
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# config_overrides = dict(args.config) if args.config else {}
# rl_config_overrides = dict(args.rl_config) if args.rl_config else {}

# ckpt_path = epath.Path(__file__).parent / "checkpoints" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
# ckpt_path.mkdir(parents=True, exist_ok=True)

# np.set_printoptions(precision=3, suppress=True, linewidth=100)

# env_name = args.env
# env = registry.load(env_name, config_overrides=config_overrides)
# env_cfg = registry.get_default_config(env_name)
# ppo_params = locomotion_params.brax_ppo_config(env_name)
# ppo_params.update(rl_config_overrides)


# if args.wandb:
#   import wandb
#   run = wandb.init(project="mujoco_playground", 
#                    entity="bitbots",
#                    name=args.run_name,
#                    #dir=ckpt_path,
#                    config={
#                           "env": args.env,
#                           "gpu": args.gpu,
#                           "run_name": args.run_name,} | dict(ppo_params) | dict(env_cfg) | config_overrides,)
# else:
#   from mujoco_playground.experimental.utils.plotting import TrainingPlotter
#   plotter = TrainingPlotter(max_timesteps=ppo_params.num_timesteps, figsize=(15, 10))

# x_data, y_data, y_dataerr = [], [], []
# times = [datetime.now()]

# pbar = tqdm(total=ppo_params.num_timesteps, desc="Training Progress", unit="steps", dynamic_ncols=True)

# def progress(num_steps, metrics):
#   times.append(datetime.now())
#   if args.wandb:
#     metrics_name_replaced = {k.replace("eval/episode_reward/", "rewards/"): v for k, v in metrics.items()}
#     for k, v in metrics.items():
#       if "rewards" in k and "_std" in k:
#         new_k = k.replace("rewards", "rewards_std")
#         print(f'replace {k} with {new_k}')
#         del(metrics_name_replaced[k])
#         metrics_name_replaced[new_k] = v
#     wandb.log(metrics_name_replaced, step=num_steps)
#   else:
#     plotter.update(num_steps, metrics)
#     plotter.save_figure(ckpt_path / f"ppo_training_progress_plots_{num_steps:012}.png")
#     x_data.append(num_steps)
#     y_data.append(metrics["eval/episode_reward"])
#     y_dataerr.append(metrics["eval/episode_reward_std"])
#     fig, ax = plt.subplots(1,1)
#     ax.set_ylim([0, ppo_params["num_timesteps"] * 1.25])
#     ax.set_xlabel("# environment steps")
#     ax.set_ylabel("reward per episode")
#     ax.set_title(f"y={y_data[-1]:.3f}")
#     ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
#     fig.savefig(ckpt_path / f"ppo_training_progress_{num_steps:012}.png", dpi=300, bbox_inches='tight')
#   pbar.update(num_steps - pbar.n)

# randomizer = registry.get_domain_randomizer(env_name)
# ppo_training_params = dict(ppo_params)


# network_factory = ppo_networks.make_ppo_networks
# if "network_factory" in ppo_params:
#   del ppo_training_params["network_factory"]
#   network_factory = functools.partial(
#       ppo_networks.make_ppo_networks,
#       **ppo_params.network_factory
#   )



# def policy_params_fn(current_step, make_policy, params):
#   del make_policy  # Unused.
#   save_params(ckpt_path, params, current_step)

# train_fn = functools.partial(
#     ppo.train, **dict(ppo_training_params),
#     network_factory=network_factory,
#     randomization_fn=randomizer,
#     progress_fn=progress,
#     #save_checkpoint_path=checkpoint_path,
#     policy_params_fn=policy_params_fn,
# )
# make_inference_fn, params, metrics = train_fn(
#     environment=env,
#     eval_env=registry.load(env_name, config=env_cfg, config_overrides=config_overrides),
#     wrap_env_fn=wrapper.wrap_for_brax_training,
# )
# print(f"time to jit: {times[1] - times[0]}")
# print(f"time to train: {times[-1] - times[1]}")

# del env

def save_params(ckpt_path, params, step=-1):
    normalizer_params, policy_params, value_params = params
    filename = ckpt_path / (f"params_{step:012}.pkl" if step >= 0 else "params.pkl")
    with open(filename, "wb") as f:
        data = {
            "normalizer_params": normalizer_params,
            "policy_params": policy_params,
            "value_params": value_params,
        }
        pickle.dump(data, f)
    print("Brax PPO Parameter gespeichert:", filename)


def render_video_brax(env_name, env_cfg, params, make_inference_fn, config_overrides_dict, seed):
    print("Rendering Video")
    import mujoco
    import mediapy as media
    import functools
    from mujoco_playground._src.gait import draw_joystick_command

    # Eval Environment
    eval_env = registry.load(
        env_name,
        config=env_cfg,
        config_overrides=config_overrides_dict,
    )
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(seed)

    rollout = []
    modify_scene_fns = []

    commands = jp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, -0.4, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, -2.0],
    ])

    phase_dt = 2 * jp.pi * eval_env.dt * 1.5
    phase = jp.array([0, jp.pi])

    for j in range(commands.shape[0]):
        print(f"episode {j}")
        state = jit_reset(rng)
        state.info["phase_dt"] = phase_dt
        state.info["phase"] = phase

        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)

            if state.done:
                break

            state.info["command"] = commands[j]
            rollout.append(state)

            torso_body_name = "pelvis_link"
            xyz = np.array(state.data.xpos[eval_env.mj_model.body(torso_body_name).id])
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
        height=480 * 2,
        modify_scene_fns=mod_fns,
    )

    out_path = f"{env_name}_brax_eval.mp4"
    media.write_video(out_path, frames, fps=fps)
    print("Video gespeichert unter:", out_path)


def main(
    argv,
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
    """Brax Joystick Training Einstieg, wird von Optuna und vom CLI Wrapper genutzt."""

    print("Joystick main aufgerufen")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = epath.Path(__file__).parent / "checkpoints" / f"{timestamp}_{run_name}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    print("env_name :", env_name)
    print("run_name :", run_name)
    print("device   :", device)
    print("use_wandb:", use_wandb)
    print("seed     :", seed)

    # Hilfsfunktion fuer CLI Aufruf mit Strings
    def kv_list_to_dict(maybe_list):
        if maybe_list is None:
            return {}
        if isinstance(maybe_list, dict):
            return dict(maybe_list)
        result = {}
        for item in maybe_list:
            key, value = parse_kv(item)
            result[key] = value
        return result

    if config_is_string:
        config_overrides_dict = kv_list_to_dict(config_overrides)
        rl_config_overrides_dict = kv_list_to_dict(rl_config_overrides)
    else:
        config_overrides_dict = dict(config_overrides)
        rl_config_overrides_dict = dict(rl_config_overrides)

    print("Erzeuge Environment und PPO Konfiguration")

    env = registry.load(env_name, config_overrides=config_overrides_dict)
    env_cfg = registry.get_default_config(env_name)

    # rl_config_overrides kommt bei Optuna als komplettes Dict der Brax PPO Konfiguration
    ppo_params = config_dict.ConfigDict(rl_config_overrides_dict)

    #Für den Entwicklungsmodus habe ich hier kleinere werte eingefügt. Später die 5 zeilen entfernen
    ppo_params.num_timesteps = int(2e6)
    ppo_params.num_evals = 2
    ppo_params.num_envs = 1024
    print("DEV Override num_timesteps:", ppo_params.num_timesteps)
    print("DEV Override num_envs:", ppo_params.num_envs)
    #####################################
    #print("Env Zeitschritt dt:", env.dt)
    #print("PPO num_timesteps:", ppo_params.num_timesteps)


    # Brax Train Parameter vorbereiten
    randomizer = registry.get_domain_randomizer(env_name)
    ppo_training_params = dict(ppo_params)

    # Seed explizit setzen, damit jede Optuna Trial reproduzierbar ist
    ppo_training_params["seed"] = seed

    # Netzwerke fuer Brax PPO bauen
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory,
        )

    print("Starte Training mit Brax PPO")

    train_fn = functools.partial(
        ppo.train,
        **ppo_training_params,
        network_factory=network_factory,
        randomization_fn=randomizer,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load(
            env_name,
            config=env_cfg,
            config_overrides=config_overrides_dict,
        ),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    # Auswertung fuer Optuna
    mean_reward = float(metrics.get("eval/episode_reward", 0.0))
    mean_episode_length = float(
        metrics.get("eval/episode_length", env_cfg.episode_length)
    )

    print("Training fertig")
    save_params(ckpt_path, params)
    print("Env Zeitschritt dt:", env.dt)
    print("PPO num_timesteps:", ppo_params.num_timesteps)

    
    if run_eval:
        print("Starte Auswertung mit Joystick Kommandos")

        # Eval Environment
        eval_env = registry.load(
            env_name,
            config=env_cfg,
            config_overrides=config_overrides_dict,
        )
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)

        # Inferenz Funktion aus dem trainierten Brax PPO
        jit_inference_fn = jax.jit(
            make_inference_fn(params, deterministic=True)
        )

        rng = jax.random.PRNGKey(seed)

        # Gleiche Kommandos wie in der alten Brax Version und in RSL RL
        commands = jp.array([
            [0.0,  0.0,  0.0],
            [1.0,  0.0,  0.0],
            [-0.5, 0.0,  0.0],
            [0.0,  0.4,  0.0],
            [0.0, -0.4,  0.0],
            [0.0,  0.0,  2.0],
            [0.0,  0.0, -2.0],
        ])

        phase_dt = 2 * jp.pi * eval_env.dt * 1.5
        base_phase = jp.array([0.0, jp.pi])

        sum_reward = 0.0
        sum_episode_length = 0

        for j in range(commands.shape[0]):
            print(f"Eval Episode {j}")
            state = jit_reset(rng)

            # Zusätzliche Infos wie im alten Joystick Code
            state.info["phase_dt"] = phase_dt
            state.info["phase"] = base_phase

            for t in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)

                # Joystick Kommando in state.info eintragen
                state.info["command"] = commands[j]

                # Reward und Episodenlänge aufsummieren
                sum_reward += float(state.reward)
                sum_episode_length += 1

                if state.done:
                    break

        mean_reward = sum_reward / float(commands.shape[0])
        mean_episode_length = sum_episode_length / float(commands.shape[0])

    #hier wird die render methode aufgerufen
    if render_video:
        render_video_brax(
            env_name=env_name,
            env_cfg=env_cfg,
            params=params,
            make_inference_fn=make_inference_fn,
            config_overrides_dict=config_overrides_dict,
            seed=seed,
        )
    #Export der Checkpoints    
    if export_onnx:
        print("Exportiere ONNX Modell")
        param_file = ckpt_path / "params.pkl"
        onnx_out = ckpt_path / f"{run_name}.onnx"
        conv_to_onnx(param_file, onnx_out, env_name)
        print("ONNX gespeichert:", onnx_out)

    print("Eval mean_reward:", mean_reward)
    print("Eval mean_episode_length:", mean_episode_length)

    print("Training fertig")
    print("mean_reward:", mean_reward)
    print("mean_episode_length:", mean_episode_length)

    return mean_reward, mean_episode_length

if __name__ == "__main__":
    _ENV_NAME = flags.DEFINE_string("env_name", "X02JoystickFlatTerrain", "Name of the environment")
    _RUN_NAME = flags.DEFINE_string("run_name", None, "Name of the run")
    _DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training")
    _USE_WANDB = flags.DEFINE_boolean("wandb", False, "Enable Weights and Biases logging")
    _CONFIG_OVERRIDES = flags.DEFINE_spaceseplist("config_overrides", [], "Config overrides")
    _RL_CONFIG_OVERRIDES = flags.DEFINE_spaceseplist("rl_config_overrides", [], "RL config overrides")
    _SEED = flags.DEFINE_integer("seed", 1, "Random seed")
    _RENDER_VIDEO = flags.DEFINE_boolean("render_video", False, "Render video")

    def main_wrapper(argv):
        return main(
            argv,
            env_name=_ENV_NAME.value,
            run_name=_RUN_NAME.value,
            device=_DEVICE.value,
            config_overrides=_CONFIG_OVERRIDES.value,
            rl_config_overrides=_RL_CONFIG_OVERRIDES.value,
            use_wandb=_USE_WANDB.value,
            render_video=_RENDER_VIDEO.value,
            run_eval=False,
            export_onnx=True,
            config_is_string=True,
        )

    app.run(main_wrapper)