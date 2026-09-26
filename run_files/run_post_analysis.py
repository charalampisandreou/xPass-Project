import os
import sys
import glob
from datetime import datetime
import pandas as pd
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_analysis_pipeline import player_analysis, team_analysis

def extract_tags(csv_path: str) -> str:
    """
    Extracts the league/season identifier from input filenames.
    """
    filename = os.path.basename(csv_path)

    base = os.path.splitext(filename)[0]

    if base.startswith("xp_added_passes_"):
        base = base[len("xp_added_passes_"):]

    #Strip datetime pattern
    cleaned = re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', base)

    return cleaned if cleaned else "all_matches"



def main():
    print("Starting Post-Analysis Processing...\n")

    # Load the latest processed dataset
    csv_pattern = os.path.join(PROJECT_ROOT, "models", "final", "*", "datasets", "pre analysis", "xp_added_passes_*.csv")

    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        csv_pattern = os.path.join(PROJECT_ROOT, "data", "processed", "xp_added_passes_*.csv")
        csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print("Error: No processed pass datasets found in models/final or data/processed directory.")
        return

    latest_csv = max(csv_files, key = os.path.getctime)
    print(f"Loading the latest dataset: {latest_csv}")
    df = pd.read_csv(latest_csv)
    print(f"Dataset successfully loaded: {len(df)} passes across {len(df.columns)} columns.")

    # Perform player analysis
    print("\nPerforming player analysis...")
    player_df = player_analysis(df)

    # Perform team analysis
    print("\nPerforming team analysis...")
    team_df = team_analysis(df)

    # Save the results
    print("\nSaving the post analysis datasets...")

    pre_analysis_dir = os.path.dirname(latest_csv)
    model_dataset_dir = os.path.dirname(pre_analysis_dir)
    output_dir = os.path.join(model_dataset_dir, "post analysis")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    status_tag = extract_tags(latest_csv)
    player_output_path = os.path.join(output_dir, f"player_analysis_{status_tag}_{date_str}.csv")
    team_output_path = os.path.join(output_dir, f"team_analysis_{status_tag}_{date_str}.csv")

    player_df.to_csv(player_output_path, index = False)
    print(f"Player analysis dataset saved to: {player_output_path}")

    team_df.to_csv(team_output_path, index = False)
    print(f"Team analysis dataset saved to: {team_output_path}")

    print("\n[SUCCESS]: Post-Analysis Processing Completed Successfully.")

if __name__ == "__main__":
    main()





