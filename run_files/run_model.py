"""
Step 3 · Model Training

Trains the expected-pass (xP) model on the most recent processed dataset in
data/processed. Two XGBoost classifiers are compared on a held-out test set, a plain
one and an isotonic-calibrated one, and whichever has the lower log-loss is saved.

Every run gets its own timestamped folder, so earlier models are never overwritten:

    models/final/model_<timestamp>/
        xpass_production_<timestamp>.joblib     the trained model
        feature_list.json                       the feature columns, in order, the model expects
        reports/                                metrics report, reliability and SHAP diagrams

Usage:
    python run_files/run_model.py
"""
import os
import sys
import glob
import pandas as pd
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Make the `src` package importable when this file is run directly as a script
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_training_pipeline import run_model_training
from src.features import FEAT_TRUEBEST
from src import console


def main():
    console.header("Model Training · XGBoost xPass")

    # One timestamp names both the run folder and the model file inside it
    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    model_file_str = "model_" + date_str

    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    models_dir = os.path.join(PROJECT_ROOT, "models", "final", model_file_str)

    csv_pattern = os.path.join(processed_dir, "processed_passes_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        console.error("No processed datasets found in data/processed. Run run_data_analysis.py first.")
        sys.exit(1)

    # Train on the most recently created dataset, i.e. the one Step 2 just wrote
    latest_csv = max(csv_files, key=os.path.getctime)
    console.info(f"Dataset: {os.path.basename(latest_csv)}")

    console.info("Loading passes...")
    df = pd.read_csv(latest_csv)
    console.success(f"Loaded {len(df):,} passes · {len(df.columns)} columns")

    model, is_calibrated = run_model_training(
        df = df,
        model_name = "xpass_production_" + date_str,
        models_dir = models_dir,
        force_calibration = False,
        features = FEAT_TRUEBEST
    )

    console.section("Summary")
    console.kv("Passes used", f"{len(df):,}")
    console.kv("Deployed model", "Calibrated XGBoost (isotonic)" if is_calibrated else "Raw XGBoost")
    console.kv("Saved to", os.path.relpath(models_dir, PROJECT_ROOT))

    console.done("Model training complete")


if __name__ == "__main__":
    main()
