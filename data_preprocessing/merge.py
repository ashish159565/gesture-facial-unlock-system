import pandas as pd

# Load both CSV files
df1 = pd.read_csv('data/nogesture_landmark.csv')
df2 = pd.read_csv('data/peace_landmark.csv')

# Merge them (stacking rows)
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save to a new CSV file
merged_df.to_csv('data.csv', index=False)
