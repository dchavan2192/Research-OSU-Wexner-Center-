import pandas as pd
from scipy.stats import pearsonr

# Load your data
df = pd.read_csv("dataFilePath")

df.columns = df.columns.str.strip()

# Filter out rows where 'Weight' or 'FMA Score' is "Missing" or "Removed"
filtered_df = df[(df['Weight'].notna()) & (df['FMA Score'].notna())]
filtered_df = filtered_df[(filtered_df['Weight'] != "Missing") &
                          (filtered_df['Weight'] != "Removed") &
                          (filtered_df['FMA Score'] != "Missing") &
                          (filtered_df['FMA Score'] != "Removed")].copy()

# Convert 'Weight' and 'FMA Score' to numeric
filtered_df['Weight'] = pd.to_numeric(filtered_df['Weight'], errors='coerce')
filtered_df['FMA Score'] = pd.to_numeric(filtered_df['FMA Score'], errors='coerce')

# Drop any rows with NaN values after conversion
filtered_df.dropna(subset=['Weight', 'FMA Score'], inplace=True)

# Calculate the Pearson correlation
corr, p_value = pearsonr(filtered_df['Weight'], filtered_df['FMA Score'])

print(f"Pearson correlation coefficient: {corr}")
print(f"P-value: {p_value}")
