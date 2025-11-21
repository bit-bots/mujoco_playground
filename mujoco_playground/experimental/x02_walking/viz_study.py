import optuna
import argparse
import optuna.visualization
import plotly.graph_objects as go

parser = argparse.ArgumentParser(description="Visualize Optuna study")
parser.add_argument("--study-name", type=str, required=True)
parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
args = parser.parse_args()

study = optuna.load_study(study_name=args.study_name, storage=args.storage)
print(f" There were {len(study.trials)} trials")
print(f"Best trial value: {study.best_value}")
print(f"Best trial parameters: {study.best_trial.params}")

timeline_plot : go.Figure= optuna.visualization.plot_timeline(study)
timeline_plot.write_image(f"{args.study_name}_timeline.png")

optimization_history_plot : go.Figure= optuna.visualization.plot_optimization_history(study)
optimization_history_plot.write_image(f"{args.study_name}_optimization_history.png")

param_importance_plot : go.Figure= optuna.visualization.plot_param_importances(study)
param_importance_plot.write_image(f"{args.study_name}_param_importance.png")

parallel_coordinate_plot : go.Figure= optuna.visualization.plot_parallel_coordinate(study)
parallel_coordinate_plot.write_image(f"{args.study_name}_parallel_coordinate.png")