import pandas as pd
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data from the specified file path
df = pd.read_csv("dataFilePath")

# Remove any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Filter out rows where 'FUNDING SOURCE' is "Missing" and 'FMA Score' is valid
filtered_df = df[(df['FUNDING SOURCE'] != "Missing") &
                 (~df['FUNDING SOURCE'].isna()) &
                 (df['FMA Score'] != "Missing") &
                 (df['FMA Score'] != "Removed")].copy()

# Convert 'FMA Score' to numeric, forcing non-numeric values to NaN
filtered_df['FMA Score'] = pd.to_numeric(filtered_df['FMA Score'], errors='coerce')

# Drop any rows with NaN values after conversion
filtered_df.dropna(subset=['FMA Score'], inplace=True)

# Print the count of each funding source with valid FMA scores
funding_source_counts = filtered_df['FUNDING SOURCE'].value_counts()
print("Funding Source Counts with FMA Scores:")
print(funding_source_counts)

# Perform ANOVA
groups = filtered_df.groupby('FUNDING SOURCE')['FMA Score'].apply(list)
anova_result = f_oneway(*groups)
print(f"ANOVA F-statistic: {anova_result.statistic}")
print(f"ANOVA P-value: {anova_result.pvalue}")

# Perform Tukey's HSD post-hoc test
tukey_result = pairwise_tukeyhsd(endog=filtered_df['FMA Score'], groups=filtered_df['FUNDING SOURCE'], alpha=0.05)

# Print the Tukey HSD results
print(tukey_result)

# Plot the results
tukey_result.plot_simultaneous()
plt.title('Tukey HSD Test for Funding Source vs. FMA Score')
plt.show()

# Visualize the data with a boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='FUNDING SOURCE', y='FMA Score', data=filtered_df)
plt.title('FMA Score by Funding Source')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()
