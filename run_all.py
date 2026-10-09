"""
Runs the complete xPass pipeline end to end:

    1. Data Ingestion   download StatsBomb open data (only matches not already on disk)
    2. Data Analysis    turn raw match events into one row of features per pass
    3. Model Training   train the XGBoost xP model and write its evaluation reports
    4. Predictions      calculate xP for every pass with the newest model
    5. Post-Analysis    aggregate xP and pass value added (PVA) per player and per team
    6. Visualizations   draw the leaderboard, scatter plots and pass map

Each step is a standalone script in run_files/ and runs in its own Python process, so
a crash in one step can't leave state behind for the next. The pipeline stops at the
first step that fails and prints the command to re-run just that step.

Usage:
    python run_all.py                                          # every league and season
    python run_all.py --league "La Liga" --season "2015/2016"  # a quick, single-season run
"""
import argparse
import os
import sys
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_FILES_DIR = os.path.join(PROJECT_ROOT, "run_files")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import console

# Each entry is (script, display title, whether the step accepts --league / --season).
# Training and prediction always use everything the earlier steps produced.
PIPELINE_STEPS = [
    ("run_data_ingestion.py", "Data Ingestion", True),
    ("run_data_analysis.py", "Data Analysis", True),
    ("run_model.py", "Model Training", False),
    ("run_predictions.py", "Predictions", False),
    ("run_post_analysis.py", "Post-Analysis", False),
    ("run_visualizations.py", "Visualizations", True),
]


def parse_args():
    """Reads the optional --league / --season scope from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the full xPass pipeline: download → features → train → predict → analyse → visualise.",
        epilog='Quick start: python run_all.py --league "La Liga" --season "2015/2016"',
    )
    parser.add_argument("--league", help="Limit the run to one league, e.g. \"La Liga\" (default: every competition)")
    parser.add_argument("--season", help="Limit the run to one season of --league, e.g. \"2015/2016\"")
    args = parser.parse_args()

    if args.season and not args.league:
        parser.error("--season requires --league")

    return args


def format_duration(seconds: float) -> str:
    """Formats a duration as "4.2s", "3m 07s" or "1h 02m 07s"."""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s" if minutes else f"{seconds:.1f}s"


def run_step(index: int, script_name: str, title: str, extra_args: list) -> float:
    """
    Runs one pipeline script in a child process and returns how long it took.

    If the script exits with an error, the whole pipeline stops here with the same exit code.
    """
    script_path = os.path.join(RUN_FILES_DIR, script_name)

    env = os.environ.copy()
    # XPASS_STEP lets the step print "STEP 2/6" in its header. UTF-8 keeps the console
    # symbols working on Windows, and unbuffered output keeps the parent's and child's
    # messages in the right order when the output is piped or written to a log file.
    env["XPASS_STEP"] = f"{index}/{len(PIPELINE_STEPS)}"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    result = subprocess.run([sys.executable, script_path, *extra_args], cwd=PROJECT_ROOT, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        console.banner("Pipeline Stopped")
        console.error(f"Step {index}/{len(PIPELINE_STEPS)} · {title} failed after {format_duration(elapsed)} (exit code {result.returncode}).")
        # Rebuild the exact command for this step, quoting arguments that contain spaces
        rerun = " ".join([f"python run_files/{script_name}", *(f'"{a}"' if " " in a else a for a in extra_args)])
        console.info(f"Fix the error above, then re-run this step with: {rerun}")
        sys.exit(result.returncode)

    return elapsed


def main():
    args = parse_args()

    scope_args = []
    if args.league:
        scope_args += ["--league", args.league]
    if args.season:
        scope_args += ["--season", args.season]

    console.banner("xPass · Full Pipeline")
    console.kv("Scope", console.describe_scope(args.league, args.season))
    for i, (_, title, _) in enumerate(PIPELINE_STEPS, start=1):
        console.kv("Steps" if i == 1 else "", f"{i}. {title}")

    timings = []
    pipeline_start = time.time()

    try:
        for i, (script_name, title, takes_scope) in enumerate(PIPELINE_STEPS, start=1):
            timings.append((title, run_step(i, script_name, title, scope_args if takes_scope else [])))
    except KeyboardInterrupt:
        # Ctrl+C reaches the running step as well, and it reports the interruption itself.
        # Exit code 130 is the usual convention for "stopped by Ctrl+C".
        console.banner("Pipeline Stopped")
        console.error("Interrupted by user.")
        sys.exit(130)

    total = time.time() - pipeline_start

    console.banner("Pipeline Complete")
    for i, (title, elapsed) in enumerate(timings, start=1):
        console.kv(f"{i}. {title}", format_duration(elapsed))
    console.kv("Total", format_duration(total))
    print("═" * console.WIDTH)


if __name__ == "__main__":
    main()
