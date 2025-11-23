"""
Script to load Optuna study parameters and generate command-line arguments
in config_dict format
"""
import optuna
import argparse
import sys
from typing import Dict, Any


def parse_config_dict_arg(arg: str) -> tuple:
    """Parse a config_dict path argument like 'algorithm.learning_rate=0.001'
    
    Returns:
        (path_parts, value) where path_parts is a list of strings
    """
    if '=' not in arg:
        raise ValueError(f"Argument must be in format 'path.to.param=value', got: {arg}")
    
    path_str, value_str = arg.split('=', 1)
    path_parts = path_str.split('.')
    
    # Try to convert value to appropriate type
    value = value_str
    # Try int
    try:
        if '.' not in value_str and 'e' not in value_str.lower():
            value = int(value_str)
        else:
            value = float(value_str)
    except ValueError:
        # Try bool
        if value_str.lower() == 'true':
            value = True
        elif value_str.lower() == 'false':
            value = False
        # Otherwise keep as string
    
    return path_parts, value


def set_nested_dict(d: Dict[str, Any], path_parts: list, value: Any):
    """Set a nested dictionary value given a path like ['algorithm', 'learning_rate']"""
    current = d
    for part in path_parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[path_parts[-1]] = value


def optuna_to_config_path(optuna_param: str) -> list:
    """Map Optuna parameter name to config_dict path parts"""
    mapping = {
        # Algorithm parameters
        'num_learning_epochs': ['algorithm', 'num_learning_epochs'],
        'num_mini_batches': ['algorithm', 'num_mini_batches'],
        'clip_param': ['algorithm', 'clip_param'],
        'gamma': ['algorithm', 'gamma'],
        'lam': ['algorithm', 'lam'],
        'value_loss_coef': ['algorithm', 'value_loss_coef'],
        'entropy_coef': ['algorithm', 'entropy_coef'],
        'learning_rate': ['algorithm', 'learning_rate'],
        'max_grad_norm': ['algorithm', 'max_grad_norm'],
        'use_clipped_value_loss': ['algorithm', 'use_clipped_value_loss'],
        'schedule': ['algorithm', 'schedule'],
        'desired_kl': ['algorithm', 'desired_kl'],
        
        # Training parameters
        'num_steps_per_env': ['num_steps_per_env'],
        'num_envs': ['num_envs'],
        'empirical_normalization': ['empirical_normalization'],
        
        # Policy network parameters
        'actor_hidden_dim1': ['policy', 'actor_hidden_dims', 0],  # Special handling needed
        'actor_hidden_dim2': ['policy', 'actor_hidden_dims', 1],
        'actor_hidden_dim3': ['policy', 'actor_hidden_dims', 2],
        'activation': ['policy', 'activation'],
    }
    return mapping.get(optuna_param, [optuna_param])


def main():
    parser = argparse.ArgumentParser(
        description="Load Optuna study parameters and generate config_dict command-line arguments",
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Name of the Optuna study to load"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db",
        help="Storage URL for the Optuna study (default: sqlite:///optuna_study.db)"
    )
    parser.add_argument(
        "--trial-number",
        type=int,
        default=None,
        help="Use specific trial number instead of best trial (default: use best trial)"
    )
    parser.add_argument(
        "--nth-best",
        type=int,
        default=1,
        help="Nth best trial to use (default: 1 = best trial)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["command", "dict", "json"],
        default="command",
        help="Output format: 'command' for command-line args, 'dict' for Python dict, 'json' for JSON (default: command)"
    )
    
    # Parse known args first to get study name
    args, remaining = parser.parse_known_args()
    
    # Load study
    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        print(f"Loaded study '{args.study_name}' with {len(study.trials)} trials", file=sys.stderr)
    except Exception as e:
        print(f"Error loading study: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get trial
    if args.trial_number is not None:
        trial = study.trials[args.trial_number] if args.trial_number < len(study.trials) else None
        if trial is None:
            print(f"Error: Trial number {args.trial_number} not found (study has {len(study.trials)} trials)", file=sys.stderr)
            sys.exit(1)
        print(f"Using trial {args.trial_number} (value: {trial.value})", file=sys.stderr)
        params = trial.params
    else:
        # Filter out failed trials (those with None value) and sort by value
        completed_trials = [t for t in study.trials if t.value is not None]
        selected_trial = sorted(completed_trials, key=lambda x: x.value, reverse=True)[args.nth_best - 1]
        print(f"Using {args.nth_best}th best trial (value: {selected_trial.value:.4f})", file=sys.stderr)
        params = selected_trial.params
    
    # Build config dict from Optuna parameters
    config_dict = {}
    
    # Handle actor_hidden_dims specially (it's a list from 3 separate params)
    actor_hidden_dims = []
    for i in [1, 2, 3]:
        param_name = f'actor_hidden_dim{i}'
        if param_name in params:
            actor_hidden_dims.append(params[param_name])
    
    if actor_hidden_dims:
        set_nested_dict(config_dict, ['policy', 'actor_hidden_dims'], actor_hidden_dims)
        set_nested_dict(config_dict, ['policy', 'critic_hidden_dims'], actor_hidden_dims)
    
    # Map other Optuna params to config paths
    for optuna_param, value in params.items():
        if optuna_param.startswith('actor_hidden_dim'):
            continue  # Already handled above
        
        path_parts = optuna_to_config_path(optuna_param)
        if path_parts:
            set_nested_dict(config_dict, path_parts, value)
    
    # Parse command-line overrides
    overrides = {}
    for arg in remaining:
        if '=' in arg:
            try:
                path_parts, value = parse_config_dict_arg(arg)
                set_nested_dict(overrides, path_parts, value)
            except Exception as e:
                print(f"Warning: Could not parse argument '{arg}': {e}", file=sys.stderr)
                continue
    
    # Apply overrides
    def merge_dict(base: dict, override: dict):
        """Recursively merge override dict into base dict"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value
    
    if overrides:
        merge_dict(config_dict, overrides)
        print(f"Applied {len(overrides)} override(s)", file=sys.stderr)
    
    # Output in requested format
    if args.format == "command":
        # Generate command-line arguments
        def format_value(v):
            if isinstance(v, bool):
                return str(v)
            elif isinstance(v, str):
                return v
            elif isinstance(v, (int, float)):
                return str(v)
            elif isinstance(v, list):
                return '[' + ','.join(str(x) for x in v) + ']'
            else:
                return str(v)
        
        def dict_to_args(d: dict, prefix: str = ""):
            """Convert nested dict to command-line args"""
            args_list = []
            for key, value in d.items():
                current_path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    args_list.extend(dict_to_args(value, current_path))
                else:
                    args_list.append(f"{current_path}={format_value(value)}")
            return args_list
        
        args_list = dict_to_args(config_dict)
        print(" ".join(args_list))
    
    elif args.format == "dict":
        import pprint
        pprint.pprint(config_dict)
    
    elif args.format == "json":
        import json
        print(json.dumps(config_dict, indent=2))


if __name__ == "__main__":
    main()

