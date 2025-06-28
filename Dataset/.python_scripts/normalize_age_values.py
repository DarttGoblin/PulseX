import pandas as pd

# Load your dataset (adjust the filename if needed)
df = pd.read_csv('demographic_data.csv')

# Divide the 'age' column by 100
df['Age'] = df['Age'] / 100

# Save the updated dataset
df.to_csv('normalized_dataset.csv', index=False)

print("Age column has been normalized and saved to 'normalized_dataset.csv'")