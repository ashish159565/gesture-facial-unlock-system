# balance_gesture_data.py
import pandas as pd

df = pd.read_csv("data/csv/data.csv")

# Separate classes
peace_df = df[df['label'] == 'peace']
nogesture_df = df[df['label'] == 'no_gesture']

# Downsample peace to match no_gesture
peace_sampled = peace_df.sample(n=len(nogesture_df), random_state=42)

# Combine
balanced_df = pd.concat([peace_sampled, nogesture_df], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42)  # Shuffle

balanced_df.to_csv("data/csv/balanced_gestures.csv", index=False)
