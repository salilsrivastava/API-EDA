import pandas as pd


# Load the dataset
df = pd.read_csv("heart.csv")
print(df)


# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data type information
print("\nData Types:")
print(df.dtypes)