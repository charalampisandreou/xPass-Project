# %%
import pandas as pd
import numpy as np

# Create a mock spreadsheet layout resembling a StatsBomb data split
data_check = {
    'match_id': [3788741, 3788741],
    'player': ['Enzo Fernández', 'Moises Caicedo'],
    'event_type': ['Pass', 'Pass']
}
df = pd.DataFrame(data_check)

print("Pandas version:", pd.__version__)
print("\nYour first football dataframe loaded successfully:")
display(df)

# %%
# Filter the dataframe to only show passes made by Enzo Fernández
enzo_passes = df[df['player'] == 'Enzo Fernández']

print("Filtered Dataframe:")
display(enzo_passes)

# %%
import pandas as pd

revision_df = pd.DataFrame({
    'minute': [4, 14, 22, 45, 89],
    'player': ['Enzo Fernández', 'Moises Caicedo', 'Enzo Fernández', 'Cole Palmer', 'Enzo Fernández'],
    'event_type': ['Pass', 'Tackle', 'Pass', 'Shot', 'Pass'],
    'under_pressure': [0, 1, 1, 0, 1]
})

# %%
revision_df.head()

# %%
enzo_late_events = revision_df[(revision_df['player'] == 'Enzo Fernández') & (revision_df['minute'] > 10)]
display(enzo_late_events)

# %%
high_risk_events = revision_df[(revision_df['under_pressure'] == 1) | (revision_df['minute'] > 80)]
high_risk_events

# %%
early_defensive_actions = revision_df[(revision_df['event_type'] != 'Pass') & (revision_df['minute'] < 15)]
early_defensive_actions

# %%
caicedo_under_pressure = revision_df[(revision_df['player'] == 'Moises Caicedo') & (revision_df['under_pressure'] == 1)]
caicedo_under_pressure

# %%
!pip install requests

# %%
import pandas as pd
import requests

# URL for a single match event stream from StatsBomb open data 
# (Champions League Final: Barcelona vs Juventus)
match_url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/3754058.json"

# Download the raw nested JSON data stream
raw_events = requests.get(match_url).json()

# Convert the top-level array into a base dataframe
df_raw = pd.DataFrame(raw_events)

print("Raw Dataframe Dimensions:", df_raw.shape)
print("\nFirst 3 columns of the raw file:")
print(df_raw[['id', 'type', 'player']].head(5))

# %%
df_raw['pass'].head()

# %%
df_pass = df_raw['pass'].copy()
df_pass.head()

# %%
print("Data type of the 'pass' column:", df_pass.dtype)
print("\nExample of a non-empty 'pass' entry (a dictionary):\n", df_pass.iloc[4])
print("\nExample of an empty 'pass' entry:\n", df_pass.iloc[0])

# %%
# --- The next step: Flattening the nested 'pass' data ---
# This is a critical step in your project.
# Not every event is a pass, so the 'pass' column contains many NaN values.
# We first drop these NaNs to work only with actual pass events.

pass_events_nested = df_raw['pass'].dropna()

# Now, use pd.json_normalize. This is a powerful function that turns a list
# of dictionaries (or a Series of dictionaries) into a clean DataFrame.
df_pass_flat = pd.json_normalize(pass_events_nested)

print("\nShape of the new flattened pass DataFrame:", df_pass_flat.shape)
print("\nFirst 5 rows of your new, structured pass data:")
display(df_pass_flat.head())
