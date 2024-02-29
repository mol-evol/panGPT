import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Replace this with the path to your pangenome dataset
filename = 'inputPangenome.txt'

# Read in the data
with open(filename, 'r') as file:
    data = file.readlines()

# Assuming each line is a sequence of gene families separated by whitespace
# and each gene family is a token.
tokens = [token for line in data for token in line.strip().split()]

# Calculate token frequencies
token_freq = Counter(tokens)

# Sort tokens by frequency
sorted_tokens = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)

# Define your frequency threshold
frequency_threshold = 5  # Change this to your chosen threshold

# Find the cutoff where token frequency goes below the threshold
cutoff = None
for i, (token, freq) in enumerate(sorted_tokens):
    if freq < frequency_threshold:
        cutoff = i
        break

if cutoff is not None:
    print(f"A good cutoff for vocabulary size might be {cutoff}, where tokens occur at least {frequency_threshold} times.")
else:
    print(f"No tokens found with frequency below {frequency_threshold}. Consider using the entire vocabulary.")

# For visualization, let's plot the frequency of the top N tokens
N = min(70000, len(sorted_tokens))  # Show top 70000 tokens or total number of tokens if less than 100
top_tokens = sorted_tokens[:N]
top_token_names = [token[0] for token in top_tokens]
top_token_counts = [token[1] for token in top_tokens]

plt.figure(figsize=(20, 10))
plt.bar(top_token_names, top_token_counts)
plt.title('Top Token Frequencies in Pangenome Dataset')
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.xticks(rotation=90)  # Rotate the x labels to make them readable
plt.show()
