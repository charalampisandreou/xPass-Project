import json
import os
import pandas as pd
import numpy as np

def process_match_json(file_path):
    """
    Takes a single StatsBomb JSON match file, calculates all the spatial 
    and game-state features, and returns a clean DataFrame ready for the ML model.
    """
    # Load the raw data
    with open(file_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    df = pd.DataFrame(match_data)
    
    # Grab the match ID from the file name
    match_id = os.path.basename(file_path).replace('.json', '')
    df['match_id'] = match_id
    
    # Flatten out the dictionaries to get simple text strings for the event type and the team name
    df['event_type'] = df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    df['team_name'] = df['team'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)

    # --- Match Clock Feature ---
    # Convert minutes and seconds into a single percentage from 0.0 to 1.0. 
    # Cap it at 1.0 (45 minutes) so that long injury-time periods don't skew the math.
    half_seconds = 45 * 60
    df['raw_seconds'] = (df['minute'] * 60) + df['second']

    #I will adjust the minutes so that the percentages reset for the second half
    df['adjusted_seconds'] = np.where(
        df['period'] == 2,
        df['raw_seconds'] - half_seconds,
        df['raw_seconds']
    )
    df['half_percentage'] = np.minimum(df['adjusted_seconds'] / half_seconds, 1.0)

    # --- Scoreboard Feature ---
    # Figure out the two teams playing.
    unique_teams = df['team_name'].dropna().unique()
    
    # Just in case the file is corrupted and only logged one team, default the score to 0 to prevent a crash.
    if len(unique_teams) < 2:
        df['net_score'] = 0
    else:
        team1, team2 = unique_teams[0], unique_teams[1]

        # Look into the 'shot' dictionary to see if a goal was scored
        df['shot_outcome'] = df['shot'].apply(lambda x: x.get('outcome', {}).get('name') if isinstance(x, dict) else None)

        # Flag exactly which row represents a normal goal for which team
        df['team1_goal_step'] = np.where((df['team_name'] == team1) & (df['shot_outcome'] == 'Goal'), 1, 0)
        df['team2_goal_step'] = np.where((df['team_name'] == team2) & (df['shot_outcome'] == 'Goal'), 1, 0)

        # Also check for Own Goals.
        df['team1_goal_step'] += np.where((df['team_name'] == team2) & (df['event_type'] == 'Own Goal Against'), 1, 0)
        df['team2_goal_step'] += np.where((df['team_name'] == team1) & (df['event_type'] == 'Own Goal Against'), 1, 0)

        # Keep a running tally of the score as the game progresses row by row
        df['team1_running_score'] = df['team1_goal_step'].cumsum()
        df['team2_running_score'] = df['team2_goal_step'].cumsum()

        # Calculate the goal difference from the perspective of the team currently making the pass
        conditions = [df['team_name'] == team1, df['team_name'] == team2]
        choices = [
            df['team1_running_score'] - df['team2_running_score'],
            df['team2_running_score'] - df['team1_running_score']
        ]
        df['net_score'] = np.select(conditions, choices, default=0)

    # 2. Filter down to only passing events
    df_pass = df[df['event_type'] == 'Pass'].copy()

    # Drop any passes that do not have a starting location (Data Cleaning)
    df_pass = df_pass.dropna(subset=['location'])

    # --- Target Variable (Y) ---
    # StatsBomb implicitly assumes a pass is successful if the 'outcome' is missing.
    # If it failed, it will have a reason (like 'Incomplete' or 'Out').
    # Convert this into a clean 1 (Completed) or 0 (Failed).
    df_pass['outcome_name'] = df_pass['pass'].apply(lambda x: x.get('outcome', {}).get('name') if isinstance(x, dict) else None)
    df_pass['pass_outcome'] = df_pass['outcome_name'].isna().astype(int)

    # Convert the pressure flag into a simple 1 (True) or 0 (False)
    df_pass['under_pressure'] = df_pass['under_pressure'].fillna(False).astype(int)


    # Grab the end location of the pass
    df_pass['end_location'] = df_pass['pass'].apply(lambda x: x.get('end_location') if isinstance(x, dict) else None)

    
    # Drop any passes that do not have an ending location (Data Cleaning)
    df_pass = df_pass.dropna(subset=['end_location'])

    # 3. Spatial Math
    # Split the location arrays into exact X and Y coordinates
    df_pass['start_x'] = df_pass['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
    df_pass['start_y'] = df_pass['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
    df_pass['end_x'] = df_pass['end_location'].apply(lambda x: x[0] if isinstance(x, list) else None)
    df_pass['end_y'] = df_pass['end_location'].apply(lambda x: x[1] if isinstance(x, list) else None)

    # Calculate the angle and length of the pass
    df_pass['pass_angle'] = np.arctan2(df_pass['end_y'] - df_pass['start_y'], df_pass['end_x'] - df_pass['start_x'])
    df_pass['pass_length'] = np.sqrt((df_pass['end_x'] - df_pass['start_x'])**2 + (df_pass['end_y'] - df_pass['start_y'])**2)
    
    # Calculate situational distances (The opponent's goal is always at X=120, Y=40 on this pitch)
    df_pass['dist_to_goal'] = np.sqrt((120 - df_pass['start_x'])**2 + (40 - df_pass['start_y'])**2)
    df_pass['dist_receiv_middle'] = df_pass['end_x'] - 60

    # 4. Final Cleanup
    # We generated a lot of extra columns. Let's explicitly list only the ones our ML model actually needs.
    feature_cols = [
        'period', 'half_percentage', 'net_score', 'under_pressure',
        'start_x', 'start_y', 'end_x', 'end_y', 
        'pass_angle', 'pass_length', 'dist_to_goal', 'dist_receiv_middle'
    ]

    # Create the final, clean DataFrame and reset the row numbers so they go 0, 1, 2, 3... perfectly.
    output_df = df_pass[['match_id'] + feature_cols + ['pass_outcome']].copy()
    return output_df.reset_index(drop=True)