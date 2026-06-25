import pandas as pd
import numpy as np

# Load the messy census dataset.
df = pd.read_csv('data/census.csv')
print(df.head())

# Strip all leading/trailing whitespace from column names.
df.columns = df.columns.str.strip()

# List of categorical columns.
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']

# List of quantitative columns.
numerical_cols = ['age', 'fnlgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

print("Stripping whitespaces & replacing '-' with '_' underscore for categorical column values.")
for col in df.columns:
    if col in categorical_cols:
        df[col] = df[col].str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# Number of rows that contains '?' values.
q_count = (df == '?').sum().sum()
print(f"Number of question mark '?' values: {q_count}")
    
# Replace '?' with NaN.
print("replacing '?' with NaN")
df = df.replace('?', np.nan)

# Number of null values in the dataset.
null_counts = df.isnull().sum()
print(f"Number of null vlaues in each column:")
print(null_counts)

# Address null values in the census dataset.
print("Imputing missing values with mode for categorical variables and with mean for numerical variables")
for col in df.columns:
    if col in categorical_cols and col != 'salary':
        df[col] = df[col].fillna(df[col].mode()[0])

    elif col not in categorical_cols and col != 'salary':
        df[col] = df[col].fillna(df[col].mean())

q_counts = (df == '?').sum().sum()
print(f"Number of '?' after imputation is {q_counts}.")

null_values = df.isnull().sum()
print(f"Number of null values after imputation {null_values}.")
print("Viewing the first few rows of dataset after cleaning")
print(df.head())

# Save the cleaned census data.
print("Saving the 'cleaned_census_data.csv' in data folder")
df.to_csv("data/cleaned_census_data.csv", index=False)
