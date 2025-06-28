import pandas as pd

# Load your CSV file
df = pd.read_csv('../demographic_data_survey.csv')

# Converting
df['Healthy'] = df['Healthy'].map({'No': 0, 'Yes': 1})
df['Lives'] = df['Lives'].map({'Urban': 1, 'Rural': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Smoker'] = df['Smoker'].map({'No': 0, 'Yes': 1})

# Save the updated CSV if needed
df.to_csv('../demographic_data_survey.csv', index=False)