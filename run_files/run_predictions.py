"""
Step 4 · Predictions

Adds two columns to every pass in a processed dataset:

    xP    the model's probability that the pass is completed (0 to 1)
    PVA   pass value added: actual outcome (1 or 0) minus xP, so a completed
          pass with xP 0.3 is worth +0.7 and a missed pass with xP 0.9 is worth -0.9

By default the newest model and the newest processed dataset are used. The result is
saved next to the model, so every model folder holds the data it produced:

    models/final/model_<timestamp>/datasets/pre analysis/xp_added_passes_<tag>_<timestamp>.csv

Usage:
    python run_files/run_predictions.py
    python run_files/run_predictions.py --model path/to/model.joblib --dataset path/to/passes.csv
"""
import argparse
import os
import sys
import glob
import pandas as pd
from datetime import datetime
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Make the `src` package importable when this file is run directly as a script
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictions_pipeline import calculate_xp
from src.features import FEAT_TRUEBEST
from src import console


def extract_tags(csv_path: str) -> str:
    """
    Returns the league/season part of a processed dataset's filename, so the output
    file can carry the same tag. For example:

        processed_passes_La-Liga_2015-2016_09-10-2026_17-59-10.csv  ->  "La-Liga_2015-2016"

    Falls back to "all_matches" if nothing is left after removing the prefix and timestamp.
    """
    filename = os.path.basename(csv_path)

    base = os.path.splitext(filename)[0]

    if base.startswith("processed_passes_"):
        base = base[len("processed_passes_"):]

    # Remove the trailing "_DD-MM-YYYY_HH-MM-SS" timestamp. The underscore is optional
    # because a file made without --league has nothing but the timestamp after its prefix.
    cleaned = re.sub(r'_?\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', base)

    return cleaned if cleaned else "all_matches"


def parse_args():
    """Reads the optional --model / --dataset overrides from the command line."""
    parser = argparse.ArgumentParser(description="Calculate xP for processed passes with a trained xPass model.")
    parser.add_argument("--model", help="Path to a .joblib model (default: the latest model in models/final)")
    parser.add_argument("--dataset", help="Path to a processed_passes_*.csv (default: the latest one in data/processed)")
    return parser.parse_args()


def main():
    args = parse_args()

    console.header("Predictions · xP for every pass")

    if args.model:
        model_path = args.model
    else:
        model_pattern = os.path.join(PROJECT_ROOT, "models", "final", "*", "xpass_production_*.joblib")
        model_files = glob.glob(model_pattern, recursive = True)

        if not model_files:
            console.error("No trained models found in models/final. Run run_model.py first.")
            sys.exit(1)

        model_path = max(model_files, key = os.path.getctime)

    console.info(f"Model: {os.path.basename(model_path)}")

    if args.dataset:
        csv_path = args.dataset
    else:
        csv_pattern = os.path.join(PROJECT_ROOT, "data", "processed", "processed_passes_*.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            console.error("No processed datasets found in data/processed. Run run_data_analysis.py first.")
            sys.exit(1)

        csv_path = max(csv_files, key = os.path.getctime)

    console.info(f"Dataset: {os.path.basename(csv_path)}")

    # Paths given on the command line may be mistyped, so check both before doing any work
    for path, label in ((model_path, "Model"), (csv_path, "Dataset")):
        if not os.path.isfile(path):
            console.error(f"{label} not found: {path}.")
            sys.exit(1)

    # The model is trained on FEAT_TRUEBEST, so xP must be calculated with the same features
    features = FEAT_TRUEBEST
    console.listing(features, "Features")


    status_tag = extract_tags(csv_path)

    # Save the results inside the model's own folder, next to the model that produced them
    model_folder = os.path.dirname(model_path)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_name = "xp_added_passes_" + status_tag + "_" + date_str + ".csv"
    output_path = os.path.join(model_folder, "datasets", "pre analysis", output_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    console.section("Calculating xP")
    console.info("Loading the dataset and model, then evaluating every pass...")
    df = calculate_xp(csv_path, model_path, features)
    console.success(f"Calculated xP for {len(df):,} passes")

    console.section("Saving outputs")
    console.info(f"Writing {len(df):,} rows to CSV...")
    df.to_csv(output_path, index = False)
    console.success(f"Saved xP dataset: {os.path.basename(output_path)}")

    console.section("Summary")
    console.kv("Passes with xP", f"{len(df):,}")
    console.kv("Saved to", os.path.relpath(os.path.dirname(output_path), PROJECT_ROOT))

    console.done("Predictions complete")


if __name__ == "__main__":
    main()
