import csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
"""
# Load the data (replace 'your_file.csv' with your filename)
df_prior = pd.read_csv('abi_prior_diff.csv')
data_prior = df_prior.iloc[:, 0]  # Assuming data is in the first column
df_noprior = pd.read_csv('abi_noprior_diff.csv')
data_noprior = df_noprior.iloc[:, 0]  # Assuming data is in the first column

# Define the range and bin width
min_val, max_val = -3, 3
bin_width = 0.025
bins = np.arange(min_val, max_val + bin_width, bin_width)

# Create the histogram
plt.figure(figsize=(12, 6))
plt.hist(data_prior, bins=bins, range=(min_val, max_val), edgecolor='black', alpha=0.7)

# Formatting
plt.title('Frequency Distribution (Range -3 to 3, Bin Width 0.025)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('frequency_histogram_prior.png')
"""

# Load the data
df_prior = pd.read_csv('abi_prior_diff.csv')
data_prior = df_prior.iloc[:, 0] 
df_noprior = pd.read_csv('abi_noprior_diff.csv')
data_noprior = df_noprior.iloc[:, 0]

# Define the range and bin width
min_val, max_val = -3, 3
bin_width = 0.025
bins = np.arange(min_val, max_val + bin_width, bin_width)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot Prior data
plt.hist(data_prior, bins=bins, color='skyblue', edgecolor='black', 
         alpha=0.6, label='Trained with prior loss')

# Plot No-Prior data
plt.hist(data_noprior, bins=bins, color='salmon', edgecolor='black', 
         alpha=0.6, label='Trained without prior loss')

# Formatting
plt.title('Difference Between Upsampled Encoder Output and Decoder Output')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Set x-ticks to be readable (every 0.5 units)
plt.xticks(np.arange(min_val, max_val + 0.5, 0.5))

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='upper right') # Adds the legend to distinguish colors

# Save and show
plt.tight_layout()
plt.savefig('combined_frequency_histogram.png')
plt.savefig('combined_frequency_histogram.pdf')
plt.show()