import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Input data
FILE_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/data/raw/'
FILE_NAME = 'PS_20174392719_1491204439457_log.csv'
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# Output data
OUTPUT_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/data/processed/'
OUTPUT_FILENAME = 'processed_data.parquet'
FULL_OUTPUT_PATH = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME)


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
    # df_encoded = df_encoded.drop('type', axis=1)
    return df_encoded

# Feature Engineering
def feature_engineering(df: pd.DataFrame):
    print("Performing Feature Engineering...")
    df_fe = df.copy()

    # --- 1. Translates Steps into Days and Hours ---
    df_fe['hour_of_day'] = (df_fe['step'] - 1) % 24
    df_fe['day_of_month'] = (df_fe['step'] - 1) // 24 + 1
    df_fe['day_of_week'] = (df_fe['day_of_month'] - 1) % 7 + 1

    # --- 2. Calculate Origin Balance Discrepancy ---
    df_fe['expected_newbalanceOrig'] = np.nan

    # When money leaves the account:
    df_fe = df_fe.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})
    df_fe.loc[df_fe['type'].isin(['TRANSFER', 'CASH_OUT', 'DEBIT', 'PAYMENT']), 'expected_newbalanceOrig'] = \
        df_fe['oldbalanceOrig'] - df_fe['amount']

    # When money enters the account:
    df_fe.loc[df_fe['type'].isin(['CASH_IN']), 'expected_newbalanceOrig'] = \
        df_fe['oldbalanceOrig'] + df_fe['amount']

    # Difference between actual and expected balance
    df_fe['balance_diff_orig'] = df_fe['newbalanceOrig'] - df_fe['expected_newbalanceOrig']

    # --- 3. Calculate Origin Balance Discrepancy ---
    df_fe['expected_newbalanceDest'] = np.nan

    # When money enters the account:
    df_fe.loc[(df_fe['type'].isin(['TRANSFER', 'CASH_IN', 'PAYMENT', 'DEBIT'])) & (~df_fe['nameDest'].str.startswith('M')), 'expected_newbalanceDest'] = \
        df_fe['oldbalanceDest'] + df_fe['amount']

    # When money leaves the account:
    df_fe.loc[(df_fe['type'].isin(['CASH_OUT'])) & (~df_fe['nameDest'].str.startswith('M')), 'expected_newbalanceDest'] = \
        df_fe['oldbalanceDest'] - df_fe['amount']

    # Calculate the difference between actual and expected newbalanceDest
    df_fe['balance_diff_dest'] = df_fe['newbalanceDest'] - df_fe['expected_newbalanceDest']

    # Handle NaNs
    df_fe['is_balance_diff_dest_missing'] = df_fe['balance_diff_dest'].isna().astype(int)
    df_fe['balance_diff_dest'] = df_fe['balance_diff_dest'].fillna(0)

    # --- 4. Create Binary Flag for Origin Account Emptied ---
    df_fe['emptied_orig_account'] = ((df_fe['type'].isin(['TRANSFER', 'CASH_IN'])) &
                                     (df_fe['oldbalanceOrig'] > 0) &
                                     (df_fe['newbalanceOrig'] == 0)).astype(int)

    # --- 5. Create Binary Flag for Mule Account Behavior (Destination Starts & Ends at Zero) ---
    df_fe['mule_account'] = ((df_fe['type'].isin(['TRANSFER', 'CASH_OUT']) &
                              (df_fe['oldbalanceDest'] == 0) &
                              (df_fe['newbalanceDest'] == 0) &
                              df_fe['amount'] > 0)).astype(int)

    # --- 6. Create Binary Flag for All Balances Zero with High Amount ---
    df_fe['balances_zero_with_nonzero_amount'] = ((df_fe['type'].isin(['TRANSFER', 'CASH_OUT']) &
                                                   (df_fe['oldbalanceOrig'] == 0) &
                                                   (df_fe['newbalanceOrig'] == 0) &
                                                   (df_fe['oldbalanceDest'] == 0) &
                                                   (df_fe['newbalanceDest'] == 0) &
                                                   df_fe['amount'] > 0)).astype(int)
    # --- 7. Drop Unnecessary Columns ---
    df_fe = df_fe.drop(['type', 'step', 'nameOrig', 'nameDest',
                        'expected_newbalanceOrig', 'expected_newbalanceDest'], axis=1)

    print("Features engineered successfully!")
    return df_fe

# Write preprocessed file
def write_data(df: pd.DataFrame):
    df.to_parquet(FULL_OUTPUT_PATH, engine='pyarrow')
    print(f"Wrote file to {FULL_OUTPUT_PATH}.")

if __name__ == "__main__":
    # Load data
    df = load_data()

    # Encode categorical data
    df_encoded = encoding(df)

    # Feature engineering
    df_feature_engineered = feature_engineering(df_encoded)

    # Write data
    write_data(df_feature_engineered)