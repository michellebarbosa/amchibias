import pandas as pd

file_path = "goan-stereotypes/goan-stereotypes/data/raw/stereotype_annotation.csv"
df = pd.read_csv(file_path)

# 2. Convert labels to numeric to ensure the sum works
# This assumes '1' was used for Stereotype. 
# If they used text, we map them to 1s and 0s.
annotator_cols = ['a1', 'a2', 'a3']

for col in annotator_cols:
    # This turns '1' into 1, and everything else (NaN, 0, 'Neutral') into 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# 3. Apply the Logic: Sum the votes
df['vote_count'] = df[annotator_cols].sum(axis=1)

# 4. Filter: Keep only those where 2 or more marked it as 1
accepted_stereotypes = df[df['vote_count'] >= 2].copy()

# 5. Select only the core columns for your final tuples
final_columns = ['Axis', 'Target Group', 'Attribute']
final_list = accepted_stereotypes[final_columns]

# 6. Save the final file
final_list.to_csv("accepted_tuples.csv", index=False)

print(f"Original items: {len(df)}")
print(f"Accepted items (Majority Vote): {len(final_list)}")
print("File saved as 'accepted_tuples.csv'")