import os
import sys
import glob
import pandas as pd
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_training_pipeline import run_model_training


def main():

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    model_file_str = "model_" + date_str

    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    models_dir = os.path.join(PROJECT_ROOT, "models", "final", model_file_str)

    csv_pattern = os.path.join(processed_dir, "processed_passes_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print("Error: No CSV files found in the processed directory (data/processed).")
        return

    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"Loading the latest dataset: {latest_csv}")

    df = pd.read_csv(latest_csv)
    print(f"Dataset successfully loaded: {len(df)} passes across {len(df.columns)} columns.")

    

    model, is_calibrated = run_model_training(
        df = df,
        model_name = "xpass_production_" + date_str,
        models_dir = models_dir,
        force_calibration = False,
        features = [
            'start_x',
            'start_y',
            'pass_angle',
            'dist_to_goal',
            'height_Low Pass',
            'height_High Pass',
            'body_part_Foot',
            'body_part_Head',
            'body_part_Keeper Arm',
            'play_pattern_Regular Play',
            'play_pattern_Kick Off',
            'play_pattern_Throw In',
            'play_pattern_Free Kick',
            'play_pattern_Goal Kick',
            'play_pattern_Keeper',
            'play_pattern_Corner',
            'under_pressure'
        ]
    )

    print("\nModel Training Complete!")


if __name__ == "__main__":
    main()





    