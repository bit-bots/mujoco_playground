import subprocess
import sys
from absl import app
from absl import flags
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

_NUM_EXPERIMENTS_PER_BACKLASH = flags.DEFINE_integer(
    "num_experiments_per_backlash", 3, "Number of experiments per backlash value")

_INITIAL_SEED = flags.DEFINE_integer(
    "initial_seed", 0, "Starting seed (seeds will be initial_seed .. initial_seed + num_experiments_per_backlash - 1)")

_MAX_TIME_X = flags.DEFINE_float(
    "max_timex", 10.0, "Maximum simulation time per test (in seconds)")

_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 8, "Number of parallel processes for evaluation")

_EVAL_SCRIPT = "mujoco_playground/experimental/sim2sim/evaluate_wolfgang_joystick.py"

_DISABLE_VELOCITY_LOG = flags.DEFINE_boolean(
    "disable_velocity_log", False, "Whether to disable the velocity log"
)


def _run_single(job):
    model, backlash, seed, max_time = job
    result = subprocess.run(
        [
            sys.executable, _EVAL_SCRIPT,
            "--onnx_model", str(model),
            "--random_seed", str(seed),
            "--max_time", str(max_time),
            "--backlash", str(backlash),
            "--random_backlash", "True",
            "--disable_velocity_log", str(_DISABLE_VELOCITY_LOG.value),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return job, result.stderr
    return job, None


def main(argv):
    jobs = []
    for model in ["wolfgang_grc_rand_bl.onnx", "wolfgang_grc_zero_bl.onnx"]:
        for backlash in [0.0, 0.01, 0.025, 0.05, 0.075, 0.1]:
            for seed in range(_INITIAL_SEED.value, _INITIAL_SEED.value + _NUM_EXPERIMENTS_PER_BACKLASH.value):
                jobs.append((model, backlash, seed, _MAX_TIME_X.value))

    with ProcessPoolExecutor(max_workers=_NUM_WORKERS.value) as executor:
        futures = {
            executor.submit(_run_single, job): job for job in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            job = futures[future]
            try:
                _, err = future.result()
                if err:
                    print(f"Job {job} failed:\n{err}")
            except Exception as e:
                print(f"Job {job} exception: {e}")


if __name__ == "__main__":
    app.run(main)
