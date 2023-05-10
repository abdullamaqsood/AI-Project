import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("Final Data.csv")

# Set the random seed for reproducibility
np.random.seed(123)

# Get the number of rows to remove
n_rows = int(len(data) * 0.3)

# Randomly choose which rows to remove
rows_to_remove = np.random.choice(data.index, size=n_rows, replace=False)

# Save the removed rows to a new file
removed_rows = data.loc[rows_to_remove]
removed_rows.to_csv("Final Data Testing.csv", index=False)

# Remove the chosen rows from the original DataFrame
data_updated = data.drop(rows_to_remove)

# Save the updated DataFrame to a new file
data_updated.to_csv("Final Data Training.csv", index=False)
