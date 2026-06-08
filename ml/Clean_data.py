import pandas as pd 
import numpy as np 
import logging

logging.basicConfig(level=logging.INFO)

def clean_census_data(input_path, output_path, target, test=False):
    # Load the messy census dataset.
    df = pd.read_csv(input_path)

    # Strip all leading/trailing whitespace from column names.
    df.columns = df.columns.str.strip()

    # List of categorical columns.
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']

    # List of quantitative columns.
    numerical_cols = ['age', 'fnlgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    # Strip all leading/trailing whitespace from categorical columns.
    # Convert all string values to lowercase. Replace spaces and '-' dashes with '_' underscore.
    logging.info("Stripping whitespaces & replacing '-' with '_' underscore.")
    for col in df.columns:
        if col in categorical_cols:
            df[col] = df[col].str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Number of rows that contains '?' values.
    q_count = (df == '?').sum().sum()
    logging.info("Number of question mark '?' values: {}".format(q_count))
    
    # Replace '?' with NaN.
    df = df.replace('?', np.nan)

    # Number of null values in the dataset.
    null_counts = df.isnull().sum()
    logging.info("Number of null values: {}".format(null_counts))

    # Address null values in the census dataset.
    logging.info("Imputing missing values with mode for categorical variables and with mean for numerical variables")
    for col in df.columns:
        if col in categorical_cols and col != target:
            df[col] = df[col].fillna(df[col].mode()[0])

        if col not in categorical_cols and col != target:
            df[col] = df[col].fillna(df[col].mean())

    # Save the cleaned census data.
    if test == False:
        #df.to_csv('census_cleaned.csv', index=False)
        #census_clean_df = pd.read_csv('census_cleaned.csv')
        df.to_csv(output_path, index=false)
        logging.info(f"Cleaned census data saved to {output_path}")
        return df
    if test == True:
        logging.info("Returned cleaned census data")
        return df
