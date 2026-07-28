import pandas as pd
import joblib

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

def calculate_xp(df_path: str, model_path: str, features: list = FEAT_TRUEBEST):
    df = pd.read_csv(df_path)
    model = joblib.load(model_path)

    X = df[features]
    df['xP'] = model.predict_proba(X)[:, 1]
    df['PVA'] = df['pass_outcome'] - df['xP']

    return df