"""
Step 5 · Post-Analysis

Aggregates the per-pass xP dataset into one summary table per player and one per team:
passes attempted and completed, total xP, total pass value added (PVA), actual and
expected completion rates, and completion percentage over expected (CPOE = actual - expected).

The newest xP dataset is used, and both tables are saved next to it:

    models/final/model_<timestamp>/datasets/post analysis/
        player_analysis_<tag>_<timestamp>.csv
        team_analysis_<tag>_<timestamp>.csv

Usage:
    python run_files/run_post_analysis.py
"""
import os
import sys
import glob
from datetime import datetime
import pandas as pd
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Make the `src` package importable when this file is run directly as a script
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_analysis_pipeline import player_analysis, team_analysis
from src import console

def extract_tags(csv_path: str) -> str:
    """
    Returns the league/season part of an xP dataset's filename, so the summary files
    can carry the same tag. For example:

        xp_added_passes_La-Liga_2015-2016_09-10-2026_18-02-41.csv  ->  "La-Liga_2015-2016"

    Falls back to "all_matches" if nothing is left after removing the prefix and timestamp.
    """
    filename = os.path.basename(csv_path)

    base = os.path.splitext(filename)[0]

    if base.startswith("xp_added_passes_"):
        base = base[len("xp_added_passes_"):]

    # Remove the trailing "_DD-MM-YYYY_HH-MM-SS" timestamp. The underscore is optional
    # because a file made without --league has nothing but the timestamp after its prefix.
    cleaned = re.sub(r'_?\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', base)

    return cleaned if cleaned else "all_matches"



def main():
    console.header("Post-Analysis · player & team aggregates")

    # Find the newest xP dataset across all model folders
    csv_pattern = os.path.join(PROJECT_ROOT, "models", "final", "*", "datasets", "pre analysis", "xp_added_passes_*.csv")

    csv_files = glob.glob(csv_pattern)

    # If no model folder has one, fall back to an xP dataset placed in data/processed
    if not csv_files:
        csv_pattern = os.path.join(PROJECT_ROOT, "data", "processed", "xp_added_passes_*.csv")
        csv_files = glob.glob(csv_pattern)

    if not csv_files:
        console.error("No xP datasets found in models/final. Run run_predictions.py first.")
        sys.exit(1)

    latest_csv = max(csv_files, key = os.path.getctime)
    console.info(f"Dataset: {os.path.basename(latest_csv)}")
    console.info("Loading passes...")
    df = pd.read_csv(latest_csv)
    console.success(f"Loaded {len(df):,} passes · {len(df.columns)} columns")

    console.section("Player analysis")
    player_df = player_analysis(df)
    console.success(f"{len(player_df):,} players aggregated")

    console.section("Team analysis")
    team_df = team_analysis(df)
    console.success(f"{len(team_df):,} teams aggregated")

    console.section("Saving outputs")

    # The summaries go in ".../datasets/post analysis", alongside the "pre analysis" folder they came from
    pre_analysis_dir = os.path.dirname(latest_csv)
    model_dataset_dir = os.path.dirname(pre_analysis_dir)
    output_dir = os.path.join(model_dataset_dir, "post analysis")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    status_tag = extract_tags(latest_csv)
    player_output_path = os.path.join(output_dir, f"player_analysis_{status_tag}_{date_str}.csv")
    team_output_path = os.path.join(output_dir, f"team_analysis_{status_tag}_{date_str}.csv")

    player_df.to_csv(player_output_path, index = False)
    console.success(f"Saved player analysis: {os.path.basename(player_output_path)}")

    team_df.to_csv(team_output_path, index = False)
    console.success(f"Saved team analysis: {os.path.basename(team_output_path)}")

    console.section("Summary")
    console.kv("Players", f"{len(player_df):,}")
    console.kv("Teams", f"{len(team_df):,}")
    console.kv("Saved to", os.path.relpath(output_dir, PROJECT_ROOT))

    console.done("Post-analysis complete")

if __name__ == "__main__":
    main()
