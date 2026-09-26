import os
import sys
import glob
import pandas as pd
from datetime import datetime
import re

FEAT_TRUEBEST = [
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_tags(csv_path: str) -> str:
    """
    Extracts the league/season identifier from input filenames.
    """
    filename = os.path.basename(csv_path)

    base = os.path.splitext(filename)[0]

    if base.startswith("processed_passes_"):
        base = base[len("processed_passes_"):]

    #Strip datetime pattern
    cleaned = re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', base)

    return cleaned if cleaned else "all_matches"


from src.predictions_pipeline import calculate_xp

def main():
    print("Starting xPass Processing...\n")

    model_pattern = os.path.join(PROJECT_ROOT, "models", "final", "*", "xpass_production_*.joblib")
    print("Searching in:", os.path.join(PROJECT_ROOT, "models", "final"))
    model_files = glob.glob(model_pattern, recursive = True)

    if not model_files:
        print("Error: No model files found in the models/final directory.")
        return

    print("Do you want the latest created model? [Y/N]")
    a = input().strip().upper()

    if(a == 'Y'):
        model_path = max(model_files, key = os.path.getctime)
    else:
        print('Enter model directory: ')
        model_path = input().strip().replace("'", "").replace('"', "")

    print(f"Loading the model: {model_path}")

    csv_pattern = os.path.join(PROJECT_ROOT, "data", "processed", "processed_passes_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("Error: No processed pass datasets found in data/processed directory.")
        return
    
    print("\nDo you want the latest created dataset? [Y/N]")
    b = input().strip().upper()
    
    if(b == 'Y'):
        csv_path = max(csv_files, key = os.path.getctime)
    else:
        print('Enter dataset directory: ')
        csv_path = input().strip().replace("'", "").replace('"', "")
    
    print(f"Loading the dataset: {csv_path}")

    print(f"\nWould you like to use the FEAT_TRUEBEST features?\n{FEAT_TRUEBEST}\n[Y/N]")
    c = input().strip().upper()

    if(c == 'Y'):
        features = FEAT_TRUEBEST
    else:
        print("\nEnter features (when finished enter 0 and enter): ")
        features = []
        while True:
            new_feature = input().strip().replace("'", "").replace('"', "")
            if new_feature == '0':
                break
            else:
                features.append(new_feature)


    status_tag = extract_tags(csv_path)
    
    model_folder = os.path.dirname(model_path)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_name = "xp_added_passes_" + status_tag + "_" + date_str + ".csv"
    output_path = os.path.join(model_folder, "datasets", "pre analysis", output_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = calculate_xp(csv_path, model_path, features)

    df.to_csv(output_path, index = False)

    print(f"\n[SUCCESS]: Enriched dataset saved to: {output_path}")


if __name__ == "__main__":
    main()

    