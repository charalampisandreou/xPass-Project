import glob
import os
import sys
import pandas as pd
from datetime import datetime
import multiprocessing as mp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_analysis_pipeline import process_match_json


def process_file(file_path):
    """
    Same body as your try/except loop, just moved into a function so each
    process can run it independently. Returns (file_name, match_df, error).
    """
    file_name = os.path.basename(file_path)
    try:
        match_df = process_match_json(file_path)
        return file_name, match_df, None
    except Exception as e:
        return file_name, None, str(e)

def clean_league(val: str) -> str:
    """Replaces spaces, slashes, and quotes with clean underscores for paths."""
    return val.strip().replace("'", "").replace('"', "").replace("-", " ")

def clean_season(val: str) -> str:
    return val.strip().replace("'", "").replace('"', "").replace("/", "-").replace(" ", "-").replace("_", "-")



def main():
    json_directory = os.path.join(PROJECT_ROOT, "data", "raw", "statsbomb_free")
    output_directory = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(output_directory, exist_ok=True)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    print("\nWould you like to process every match available? [Y/N]")
    a = input().strip().upper()

    if(a == 'Y'):
        print("Processing every match available...")
        output_path = os.path.join(output_directory, f"processed_passes_{date_str}.csv")
        json_pattern = os.path.join(json_directory, "**", "*.json")

    else:
        print("\nEnter League to process: ")
        league_name = clean_league(input())
        league_name_form = league_name.replace(" ", "-")


        print("\nWould you like to process a specific season? [Y/N]")
        b = input().strip().upper()

        if(b == 'Y'):
            print("\nEnter season to process: ")
            season_name = clean_season(input())
            print(f"Processing {league_name} - {season_name}...")
            output_path = os.path.join(output_directory, f"processed_passes_{league_name_form}_{season_name}_{date_str}.csv")
            json_pattern = os.path.join(json_directory, league_name, season_name, "*.json")

        else:
            print(f"Processing every season available of {league_name}...")
            output_path = os.path.join(output_directory, f"processed_passes_{league_name_form}_all_seasons_{date_str}.csv")
            json_pattern = os.path.join(json_directory, league_name, "**", "*.json")



        

    json_files = glob.glob(json_pattern, recursive=True)

    if not json_files:
        print("\nFailed: No JSON files matched the specified pattern. Double-check your folder names.")
        return

    print(f"Found {len(json_files)} match JSON files to process.")

    processed_matches_list = []

    # Use all cores except one, so the multiprocessing runs don't freeze your computer
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Using {num_cores} processes.")

    with mp.Pool(num_cores) as pool:
        for file_name, match_df, error in pool.imap_unordered(process_file, json_files):

            if error:
                print(f"Error processing {file_name}: {error}")

            elif isinstance(match_df, pd.DataFrame) and not match_df.empty:
                processed_matches_list.append(match_df)
                print(f"Successfully processed: {file_name} with {len(match_df)} rows")

            else:
                print(f"Warning: {file_name} returned empty or invalid data. Skipping.")

    if processed_matches_list:
        master_df = pd.concat(processed_matches_list, ignore_index=True)
        master_df.to_csv(output_path, index=False)

        print("\n" + "="*50)
        print("PIPELINE CONSOLIDATION COMPLETE!")
        print(f"Matches successfully aggregated: {len(processed_matches_list)}/{len(json_files)}")
        print(f"Total passes extracted:          {len(master_df)}")
        print(f"File saved to:                   {output_path}")
        print("="*50)

    else:
        print("\nFailed: No data was successfully processed. Check your paths.")


if __name__ == '__main__':
    main()