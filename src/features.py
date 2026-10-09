"""
The feature set the xPass model is trained on. Training and prediction both import it
from here, so the two can never drift apart.

All columns are produced by src/data_analysis_pipeline.py. The categorical features are
one-hot encoded with a baseline left out of each group: ground passes for height, and any
body part or play pattern without its own column (e.g. "From Counter") for the other two.
"""
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
