import optuna
import os
os.environ['MUJOCO_GL'] = 'disable' 
import argparse
from functools import partial

import jax
from ml_collections import config_dict

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

from mujoco_playground.experimental.x02_walking.train_x02_walk_joystick import main
from mujoco_playground.config import locomotion_params

# Global variable to store device (set via command-line argument)

def objective(trial : optuna.Trial, device: str="cuda:0", env_name: str="BerkeleyHumanoidJoystickFlatTerrain"):
    """Optuna objective function for hyperparameter optimization.
    
    Returns the mean episode reward (to maximize).
    """
    print(f"Using device: {device}")
    
    # Algorithm hyperparameters
    num_updates_per_batch = trial.suggest_int("num_updates_per_batch", 1, 8)
    num_minibatches = trial.suggest_categorical("num_minibatches", [32, 64])
    discounting = trial.suggest_float("discounting", 0.9, 0.99)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.2, 1.0)
    num_evals= trial.suggest_int ("num_evals", 5, 40)
    reward_scaling = trial.suggest_float("reward_scaling", 0.1, 2)

    action_repeat= trial.suggest_int("action_repeat", 1, 3)
    unroll_length=trial.suggest_int("unroll_length", 20, 60)
    entropy_cost= trial.suggest_float("entropy_cost", 1e-4, 1e-1)
    batch_size= trial.suggest_categorical("batch_size", [256, 512, 1024])
    
    # policy Network hyperparameters
    num_policy_layers = trial.suggest_int("num_policy_layers", 2, 5)
    policy_layer_size = trial.suggest_int("policy_layer_size", 64, 256)
    policy_hidden = tuple([policy_layer_size] * num_policy_layers)
    
    num_value_layers = trial.suggest_int("num_value_layers", 3, 7)
    value_layer_size = trial.suggest_int("value_layer_size", 100, 512)
    value_hidden = tuple([value_layer_size] * num_value_layers)
    network_factory = config_dict.create(
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
        policy_obs_key="state",
        value_obs_key="state")

    # Training hyperparameters
    num_steps_per_env = trial.suggest_int("num_steps_per_env", 16, 64, step=8)
    num_envs = trial.suggest_categorical("num_envs", [8192])
    
    # Update config with suggested values
    base_config = locomotion_params.brax_ppo_config(env_name)
    base_config.num_updates_per_batch = num_updates_per_batch
    base_config.num_minibatches = num_minibatches
    base_config.discounting = discounting
    base_config.learning_rate = learning_rate
    base_config.max_grad_norm = max_grad_norm
    base_config.num_evals = num_evals
    base_config.reward_scaling = reward_scaling
    base_config.action_repeat = action_repeat
    base_config.unroll_length = unroll_length
    base_config.entropy_cost = entropy_cost
    base_config.batch_size = batch_size
    base_config.network_factory = network_factory

    base_config.num_envs = num_envs
    
    # Set training parameters for optimization (shorter training for faster trials)
    run_name = f"optuna_trial_{trial.number}"
    #base_config.experiment_name = "optuna_optimization"
    base_config.seed = 1

    
    try:
        mean_reward, mean_episode_length = main(
            argv=[],
            env_name=env_name,
            run_name=run_name,
            device=device,
            config_overrides=dict(),
            rl_config_overrides=base_config.to_dict(),
            use_wandb=False,
            seed=base_config.seed,
            render_video=False,
            run_eval=True,
            export_onnx=True,
            config_is_string=False,
        )
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        # prin traceback
        import traceback
        traceback.print_exc()
        return None
    
    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for Brax training")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'cpu'). "
            "If not specified, defaults to 'cuda:0' if CUDA is available, else 'cpu'."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials to run (default: 50)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="x02_hyperparameter_optimization",
        help="Name of the Optuna study (default: x02_hyperparameter_optimization)"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="BerkeleyHumanoidJoystickFlatTerrain",
        help="Name of the environment (default: BerkeleyHumanoidJoystickFlatTerrain)"
    )
    args = parser.parse_args()
    study = optuna.create_study(
        direction="maximize",  # Maximize mean episode reward
        study_name=args.study_name,
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )
    print(f"Starting optimization with {args.n_trials} trials...")
    partial_objective = partial(objective, device=args.device, env_name=args.env_name)
    study.optimize(partial_objective, n_trials=args.n_trials)
    
    # Print best results
    print("\nBest trial:")
    print(f"  Value (mean reward): {study.best_value:.4f}")
    print(f"  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    

