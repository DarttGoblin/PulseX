import pandas as pd

# Define class name and input path
class_name = "normal"
input_path = f"../statistical_features/{class_name}_statistical_features.csv"

# Load CSV and compute averages
df = pd.read_csv(input_path)
averages = df.drop(columns=["file_name"]).mean(numeric_only=True)

# Add class name as a column
averages["class"] = class_name

# Reorder columns
averages = averages[["class"] + [col for col in averages.index if col != "class"]]

# Convert to DataFrame (single row)
averages_df = pd.DataFrame([averages])

# Append or create final CSV
output_path = "statistical_features_averages.csv"
try:
    existing_df = pd.read_csv(output_path)
    final_df = pd.concat([existing_df, averages_df], ignore_index=True)
except FileNotFoundError:
    final_df = averages_df

# Save result
final_df.to_csv(output_path, index=False)
print(f"Averages for '{class_name}' saved to {output_path}")
