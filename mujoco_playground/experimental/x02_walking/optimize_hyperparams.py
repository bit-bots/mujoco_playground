import optuna
import os
os.environ['MUJOCO_GL'] = 'osmesa' 
import argparse
from functools import partial

import jax

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

from mujoco_playground.experimental.x02_walking.train_x02_walk_joystick_rsl import main
from mujoco_playground.config import locomotion_params

# Global variable to store device (set via command-line argument)

def objective(trial, device: str="cuda:0", env_name: str="BerkeleyHumanoidJoystickFlatTerrain"):
    """Optuna objective function for hyperparameter optimization.
    
    Returns the mean episode reward (to maximize).
    """
    print(f"Using device: {device}")
    
    # Algorithm hyperparameters
    num_learning_epochs = trial.suggest_int("num_learning_epochs", 1, 8)
    num_mini_batches = trial.suggest_int("num_mini_batches", 1, 64, log=True)
    clip_param = trial.suggest_float("clip_param", 0.1, 0.3)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    lam = trial.suggest_float("lam", 0.9, 0.99)
    value_loss_coef = trial.suggest_float("value_loss_coef", 0.5, 1.0)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.01, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.2, 1.0)
    use_clipped_value_loss = trial.suggest_categorical("use_clipped_value_loss", [True, False])
    schedule = trial.suggest_categorical("schedule", ["adaptive", "fixed"])
    desired_kl = trial.suggest_float("desired_kl", 0.003, 0.03, log=True)

    # Training hyperparameters
    num_steps_per_env = trial.suggest_int("num_steps_per_env", 16, 64, step=8)
    num_envs = trial.suggest_categorical("num_envs", [8192])
    #max_iterations = trial.suggest_categorical("max_iterations", [500])
    empirical_normalization = trial.suggest_categorical("empirical_normalization", [True, False])
    
    # Policy network hyperparameters
    network_hidden_dim1 = trial.suggest_categorical("actor_hidden_dim1", [256, 512, 1024])
    network_hidden_dim2 = trial.suggest_categorical("actor_hidden_dim2", [128, 256, 512])
    network_hidden_dim3 = trial.suggest_categorical("actor_hidden_dim3", [64, 128, 256])
    activation = trial.suggest_categorical("activation", ["elu", "relu", "tanh"])
    
    # Update config with suggested values
    base_config = locomotion_params.rsl_rl_config(env_name)
    base_config.algorithm.num_learning_epochs = num_learning_epochs
    base_config.algorithm.num_mini_batches = num_mini_batches
    base_config.algorithm.clip_param = clip_param
    base_config.algorithm.gamma = gamma
    base_config.algorithm.lam = lam
    base_config.algorithm.value_loss_coef = value_loss_coef
    base_config.algorithm.entropy_coef = entropy_coef
    base_config.algorithm.learning_rate = learning_rate
    base_config.algorithm.max_grad_norm = max_grad_norm
    base_config.algorithm.use_clipped_value_loss = use_clipped_value_loss
    base_config.algorithm.schedule = schedule
    base_config.algorithm.desired_kl = desired_kl

    base_config.num_steps_per_env = num_steps_per_env
    base_config.num_envs = num_envs
    base_config.max_iterations = 100_000_000 // (num_envs * num_steps_per_env) // 100
    base_config.empirical_normalization = empirical_normalization
    
    base_config.policy.actor_hidden_dims = [network_hidden_dim1, network_hidden_dim2, network_hidden_dim3]
    base_config.policy.critic_hidden_dims = [network_hidden_dim1, network_hidden_dim2, network_hidden_dim3]
    base_config.policy.activation = activation
    
    # Set training parameters for optimization (shorter training for faster trials)
    base_config.seed = 1
    base_config.run_name = f"optuna_trial_{trial.number}"
    base_config.experiment_name = "optuna_optimization"
    
    try:
        mean_reward, mean_episode_length = main(
            argv=[],
            env_name=env_name,
            run_name=base_config.run_name,
            device=device,
            config_overrides=dict(),
            rl_config_overrides=base_config.to_dict(),
            use_wandb=False,
            seed=base_config.seed,
            render_video=False,
            run_eval=True,
            export_onnx=False,
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
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for RSL-RL training")
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
    
