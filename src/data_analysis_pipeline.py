"""
Feature engineering: turns one StatsBomb match file into one row per pass.

StatsBomb coordinates are in yards on a 120 x 80 pitch, with the team in possession
always attacking towards x = 120. The opponent's goal is therefore always at (120, 40).
"""
import json
import os
import pandas as pd
import numpy as np

def process_match_json(file_path):
    """
    Reads a single StatsBomb match file and returns its passes as a DataFrame.

    Each row has identifying metadata (match, team, player, recipient), the model
    features (location, angle, distance to goal, game state, pass height, body part,
    play pattern, pressure) and the target column `pass_outcome` (1 = completed, 0 = failed).
    """
    # Load the raw event list
    with open(file_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    df = pd.DataFrame(match_data)

    # Files are named <match_id>.json
    match_id = os.path.basename(file_path).replace('.json', '')
    df['match_id'] = match_id

    # Event type and team are nested objects like {"id": 30, "name": "Pass"}; keep just the name
    df['event_type'] = df['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    df['team_name'] = df['team'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)

    # --- Match clock ---
    # How far through the current half the event happened, from 0.0 to 1.0.
    # Capped at 1.0 so stoppage time doesn't push values past the end of the half.
    half_seconds = 45 * 60
    df['raw_seconds'] = (df['minute'] * 60) + df['second']

    # StatsBomb's clock keeps running into the second half (it starts at 45:00), so reset it.
    # Extra-time periods aren't reset and therefore always read as 1.0.
    df['adjusted_seconds'] = np.where(
        df['period'] == 2,
        df['raw_seconds'] - half_seconds,
        df['raw_seconds']
    )
    df['half_percentage'] = np.minimum(df['adjusted_seconds'] / half_seconds, 1.0)

    # --- Score difference ---
    # The goal difference at the moment of each event, from the acting team's point of view
    unique_teams = df['team_name'].dropna().unique()

    # A file that only records one team (incomplete data) gets a neutral score instead of crashing
    if len(unique_teams) < 2:
        df['net_score'] = 0
    else:
        team1, team2 = unique_teams[0], unique_teams[1]

        # Shot outcomes are nested inside the 'shot' object
        df['shot_outcome'] = df['shot'].apply(lambda x: x.get('outcome', {}).get('name') if isinstance(x, dict) else None)

        # Mark the events where each team scored
        df['team1_goal_step'] = np.where((df['team_name'] == team1) & (df['shot_outcome'] == 'Goal'), 1, 0)
        df['team2_goal_step'] = np.where((df['team_name'] == team2) & (df['shot_outcome'] == 'Goal'), 1, 0)

        # Own goals are logged against the team that conceded, so they count for the other side
        df['team1_goal_step'] += np.where((df['team_name'] == team2) & (df['event_type'] == 'Own Goal Against'), 1, 0)
        df['team2_goal_step'] += np.where((df['team_name'] == team1) & (df['event_type'] == 'Own Goal Against'), 1, 0)

        # Events are in match order, so a cumulative sum gives the live score
        df['team1_running_score'] = df['team1_goal_step'].cumsum()
        df['team2_running_score'] = df['team2_goal_step'].cumsum()

        # Positive when the acting team is ahead, negative when it's behind
        conditions = [df['team_name'] == team1, df['team_name'] == team2]
        choices = [
            df['team1_running_score'] - df['team2_running_score'],
            df['team2_running_score'] - df['team1_running_score']
        ]
        df['net_score'] = np.select(conditions, choices, default=0)


    # --- Passes only from here on ---
    df_pass = df[df['event_type'] == 'Pass'].copy()

    # A pass without a start location can't be placed on the pitch
    df_pass = df_pass.dropna(subset=['location'])

    # --- Target variable ---
    # StatsBomb only records an outcome for failed passes ("Incomplete", "Out", ...),
    # so a missing outcome means the pass was completed: 1 = completed, 0 = failed.
    df_pass['outcome_name'] = df_pass['pass'].apply(lambda x: x.get('outcome', {}).get('name') if isinstance(x, dict) else None)
    df_pass['pass_outcome'] = df_pass['outcome_name'].isna().astype(int)

    # under_pressure is only present when True; turn it into 1 / 0
    df_pass['under_pressure'] = df_pass['under_pressure'].fillna(False).astype(int)


    df_pass['end_location'] = df_pass['pass'].apply(lambda x: x.get('end_location') if isinstance(x, dict) else None)

    # A pass without an end location can't be measured either
    df_pass = df_pass.dropna(subset=['end_location'])

    # --- Geometry ---
    # Locations are [x, y] lists; split them into separate columns
    df_pass['start_x'] = df_pass['location'].apply(lambda x: x[0] if isinstance(x, list) else None)
    df_pass['start_y'] = df_pass['location'].apply(lambda x: x[1] if isinstance(x, list) else None)
    df_pass['end_x'] = df_pass['end_location'].apply(lambda x: x[0] if isinstance(x, list) else None)
    df_pass['end_y'] = df_pass['end_location'].apply(lambda x: x[1] if isinstance(x, list) else None)

    # Angle (radians) and length (yards) come straight from StatsBomb rather than being
    # recomputed from the coordinates with arctan2 and Pythagoras
    df_pass['pass_angle'] = df_pass['pass'].apply(lambda x: x.get('angle') if isinstance (x, dict) else None)
    df_pass['pass_length'] = df_pass['pass'].apply(lambda x: x.get('length') if isinstance (x, dict) else None)


    # Straight-line distance from where the pass starts to the centre of the opponent's goal at (120, 40)
    df_pass['dist_to_goal'] = np.sqrt((120 - df_pass['start_x'])**2 + (40 - df_pass['start_y'])**2)

    # --- Progressive passes ---
    # Based on Wyscout's definition, where a pass is progressive if it moves the ball at least
    #   30 m closer to goal when it starts and ends in the team's own half,
    #   15 m closer when it goes from the own half into the opponent's half, or
    #   10 m closer when it starts and ends in the opponent's half.
    # This version is an approximation: it compares the pass's total length (not the distance
    # gained towards goal) against those thresholds. The halfway line is x = 60.

    # StatsBomb measures in yards, so convert the metre thresholds (1 m = 1.09361 yd)
    yards_conversion = 1.09361

    cond_own = (df_pass['start_x'] < 60) & (df_pass['end_x'] < 60) & (df_pass['pass_length'] >= 30 * yards_conversion)
    cond_diff = (df_pass['start_x'] < 60) & (df_pass['end_x'] >= 60) & (df_pass['pass_length'] >= 15 * yards_conversion)
    cond_opp = (df_pass['start_x'] >= 60) & (df_pass['end_x'] >= 60) & (df_pass['pass_length'] >= 10 * yards_conversion)

    df_pass['is_progressive'] = np.select(
        [cond_own, cond_diff, cond_opp],
        [1, 1, 1],
        default = 0
    )

    # --- Categorical features: pass height, body part, play pattern ---
    df_pass['height'] = df_pass['pass'].apply(lambda x: x.get('height', {}).get('name') if isinstance(x, dict) else None)
    df_pass['body_part'] = df_pass['pass'].apply(lambda x: x.get('body_part', {}).get('name') if isinstance(x, dict) else None)
    df_pass['play_pattern'] = df_pass['play_pattern'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)

    df_pass['height'] = df_pass['height'].fillna('Ground Pass')
    df_pass['body_part'] = df_pass['body_part'].fillna('Other')
    df_pass['play_pattern'] = df_pass['play_pattern'].fillna('Other')

    # One-hot encoding is done by hand instead of with pd.get_dummies. A single match
    # doesn't always contain every category, and get_dummies would then produce different
    # columns for different matches. Listing the columns explicitly keeps every match file
    # identical. One category per group is left out as the baseline: ground passes, and
    # any body part or play pattern without its own column.

    df_pass['height_Low Pass'] = np.where(df_pass['height'] == 'Low Pass', 1, 0)
    df_pass['height_High Pass'] = np.where(df_pass['height'] == 'High Pass', 1, 0)

    df_pass['body_part_Foot'] = np.where(df_pass['body_part'].isin(['Right Foot', 'Left Foot']), 1, 0)
    df_pass['body_part_Head'] = np.where(df_pass['body_part'] == 'Head', 1, 0)
    df_pass['body_part_Keeper Arm'] = np.where(df_pass['body_part'] == 'Keeper Arm', 1, 0)

    df_pass['play_pattern_Regular Play'] = np.where(df_pass['play_pattern'] == 'Regular Play', 1, 0)
    df_pass['play_pattern_Kick Off'] = np.where(df_pass['play_pattern'] == 'From Kick Off', 1, 0)
    df_pass['play_pattern_Throw In'] = np.where(df_pass['play_pattern'] == 'From Throw In', 1, 0)
    df_pass['play_pattern_Free Kick'] = np.where(df_pass['play_pattern'] == 'From Free Kick', 1, 0)
    df_pass['play_pattern_Goal Kick'] = np.where(df_pass['play_pattern'] == 'From Goal Kick', 1, 0)
    df_pass['play_pattern_Keeper'] = np.where(df_pass['play_pattern'] == 'From Keeper', 1, 0)
    df_pass['play_pattern_Corner'] = np.where(df_pass['play_pattern'] == 'From Corner', 1, 0)


    # --- Metadata ---
    # Not used by the model, but needed to aggregate by player and team and to draw pass maps
    df_pass['event_id'] = df_pass['id']
    df_pass['team_id'] = df_pass['team'].apply(lambda x: x.get('id') if isinstance (x, dict) else None)
    df_pass['player_id'] = df_pass['player'].apply(lambda x: x.get('id') if isinstance (x, dict) else None)
    df_pass['player_name'] = df_pass['player'].apply(lambda x: x.get('name') if isinstance (x, dict) else None)
    df_pass['player_position'] = df_pass['position'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
    df_pass['pass_recipient'] = df_pass['pass'].apply(lambda x: x.get('recipient', {}).get('name') if isinstance (x, dict) else None)
    df_pass['recipient_id'] = df_pass['pass'].apply(lambda x: x.get('recipient', {}).get('id') if isinstance (x, dict) else None)

    # --- Output ---
    # Keep only the columns later steps use; the helper columns built above are dropped.
    # feature_cols is every candidate feature. The model itself uses the subset listed
    # in src/features.py.
    feature_cols = [
        'period', 'half_percentage', 'net_score', 'under_pressure',
        'start_x', 'start_y', 'end_x', 'end_y',
        'pass_angle', 'pass_length', 'dist_to_goal',
        'is_progressive',
        'height_Low Pass', 'height_High Pass',
        'body_part_Foot', 'body_part_Head', 'body_part_Keeper Arm',
        'play_pattern_Regular Play', 'play_pattern_Kick Off', 'play_pattern_Throw In',
        'play_pattern_Free Kick', 'play_pattern_Goal Kick', 'play_pattern_Keeper',
        'play_pattern_Corner'
    ]

    metadata_cols = [
        'match_id',
        'event_id',
        'team_id',
        'team_name',
        'player_name',
        'player_id',
        'player_position',
        'pass_recipient',
        'recipient_id'
    ]

    # Reset the index so the rows are numbered 0..n-1 once the non-pass events are gone
    output_df = df_pass[metadata_cols + feature_cols + ['pass_outcome']].copy()
    return output_df.reset_index(drop=True)
