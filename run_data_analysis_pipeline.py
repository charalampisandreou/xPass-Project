import glob
import os
import pandas as pd
from src.data_analysis_pipeline import process_match_json

# 1. Define paths
json_directory = "data/raw/" 
output_directory = "data/processed/"
output_path = os.path.join(output_directory, "processed_passes.csv")

# Ensure the target directory exists
os.makedirs(output_directory, exist_ok=True)

# 2. Find files
json_pattern = os.path.join(json_directory, "*.json")
json_files = glob.glob(json_pattern)

print(f"🔍 Found {len(json_files)} match JSON files to process.")

processed_matches_list = []

# 3. Iterate and process
for file_path in json_files:
    file_name = os.path.basename(file_path)
    try:
        match_df = process_match_json(file_path)
        
        if isinstance(match_df, pd.DataFrame) and not match_df.empty:
            processed_matches_list.append(match_df)
            print(f"✅ Successfully processed: {file_name} ({len(match_df)} rows)")
        else:
            print(f"⚠️ Warning: {file_name} returned empty or invalid data. Skipping.")
            
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")

# 4. Consolidate and save
if processed_matches_list:
    master_df = pd.concat(processed_matches_list, ignore_index=True)
    master_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("🚀 PIPELINE CONSOLIDATION COMPLETE!")
    print(f"Matches successfully aggregated: {len(processed_matches_list)}/{len(json_files)}")
    print(f"Total passes extracted:          {len(master_df)}")
    print(f"File saved to:                   {output_path}")
    print("="*50)
else:
    print("\n❌ Failed: No data was successfully processed. Check your paths.")