import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

file_path = "/Users/michellebarbosa/goan-stereotypes/data/raw/stereotype_annotation.csv" 
df = pd.read_csv(file_path)

# 2. Select the columns for annotators
annotator_cols = ['a1', 'a2', 'a3']
data = df[annotator_cols]

# 3. Data Cleaning: Drop any rows where an annotator missed a label
data = data.dropna()

print(f"Analyzing {len(data)} rows of annotations...")

# 4. Convert category labels (like 'Stereotype', 'Neutral') into integer codes
# Fleiss' Kappa in statsmodels requires integer input
for col in annotator_cols:
    data[col] = data[col].astype('category').cat.codes

# 5. Transform data into the format required for Fleiss' Kappa
# (Number of subjects x Number of categories)
table, _ = aggregate_raters(data.values)

# 6. Calculate Kappa
kappa = fleiss_kappa(table, method='rand')

# 7. Calculate Simple Percentage Agreement (all 3 must match)
matches = (df['a1'] == df['a2']) & (df['a2'] == df['a3'])
percent_agreement = matches.mean() * 100

# Add this to your agreement.py to see the 'problem' groups
df['is_disagreement'] = ~((df['a1'] == df['a2']) & (df['a2'] == df['a3']))
disagreement_summary = df[df['is_disagreement']]['Target Group'].value_counts()

print("\nTop Groups with Disagreement:")
print(disagreement_summary)

print("\n" + "="*30)
print(f"INTER-ANNOTATOR RESULTS")
print("="*30)
print(f"Fleiss' Kappa:      {kappa:.3f}")
print(f"Percent Agreement: {percent_agreement:.1f}%")
print("="*30)

# Interpretation
if kappa < 0: print("Interpretation: Poor Agreement")
elif kappa < 0.2: print("Interpretation: Slight Agreement")
elif kappa < 0.4: print("Interpretation: Fair Agreement")
elif kappa < 0.6: print("Interpretation: Moderate Agreement")
elif kappa < 0.8: print("Interpretation: Substantial Agreement")
else: print("Interpretation: Almost Perfect Agreement")
