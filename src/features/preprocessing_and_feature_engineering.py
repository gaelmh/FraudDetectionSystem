import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Input data
FILE_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/data/raw/'
FILE_NAME = 'PS_20174392719_1491204439457_log.csv'
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# Output data
OUTPUT_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/data/processed/'
OUTPUT_FILENAME = 'processed_data.parquet'

# Loading data
def load_data():
    """
    Loads the raw CSV data into a pandas DataFrame.
    """
    if not os.path.exists(FULL_PATH):
        print(f"Error: Data file not found at {FULL_PATH}. Please ensure it's downloaded and named correctly.")
        return None

    print(f"Loading raw data from {FULL_PATH}")
    try:
        df = pd.read_csv(FULL_PATH)
        print(f"Raw data loaded succesfully!")
        return df
    except Exception as e:
        print(f"Error loading CSV file from {FULL_PATH}: {e}")
        return None

# One Hot Encoding
def encoding(df: pd.DataFrame):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data_array = encoder.fit_transform(df[['type']])
    encoded_column_names = encoder.get_feature_names_out(['type'])
    one_hot_df = pd.DataFrame(encoded_data_array, columns=encoded_column_names, index=df.index)
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop('type', axis=1)
    df_encoded.head(5)
    return df_encoded

# Feature Engineering
def feature_engineering():
    pass

# Write preprocessed file
def write_data(df: pd.DataFrame):
    pass

if __name__ == "__main__":
    df = load_data()
    df_encoded = encoding(df)