import distutils.util
import os
import subprocess
os.environ['MUJOCO_GL'] = 'egl'  # Set the environment variable for EGL rendering
import mujoco

import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import datetime

from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
from argparse import ArgumentParser
import pickle


parser = ArgumentParser(description="Train a walking agent with joystick control.")
parser.add_argument('run_name', type=str, help='Name of the run for saving parameters')
parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU device to use')
parser.add_argument('-e', '--env', type=str, default='X02JoystickFlatTerrain')
args = parser.parse_args()

ckpt_path = epath.Path(__file__).parent / "checkpoints" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
ckpt_path.mkdir(parents=True, exist_ok=True)

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

np.set_printoptions(precision=3, suppress=True, linewidth=100)

env_name = 'X02JoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def save_params(ckpt_path, params, step=-1):
  normalizer_params, policy_params, value_params = params
  filename = ckpt_path / f"params_{step:012}.pkl" if step >= 0 else ckpt_path / "params.pkl"
  with open(filename, "wb") as f:
    data = {
      "normalizer_params": normalizer_params,
      "policy_params": policy_params,
      "value_params": value_params,
    }
    pickle.dump(data, f)

def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  plt.savefig(ckpt_path / f"ppo_training_progress_{num_steps:012}.png", dpi=300, bbox_inches='tight')

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)


network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )



def policy_params_fn(current_step, make_policy, params):
  del make_policy  # Unused.
  save_params(ckpt_path, params, current_step)

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    #save_checkpoint_path=checkpoint_path,
    policy_params_fn=policy_params_fn,
)
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

save_params(ckpt_path, params)