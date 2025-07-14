import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier

# Input data
FILE_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/data/processed/'
FILE_NAME = 'processed_data.parquet'
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# Output data
OUTPUT_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/saved_models/'
OUTPUT_FILENAME = ''
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
        df = pd.read_parquet(FULL_PATH)
        print(f"Processed data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading file from {FULL_PATH}: {e}")
        return None

# Split data
def split_data(data: pd.DataFrame):
    print("\nStarting Test-train Split...")
    # Time-based split calculation
    day_of_month_range = pd.Series(data['day_of_month'].unique())
    day_split = np.ceil(day_of_month_range.quantile(0.7))

    # Train-test Split
    train_set = data[data['day_of_month'] <= day_split]
    test_set = data[data['day_of_month'] > day_split]

    # Target and features split
    train_X = train_set.drop('isFraud', axis=1)
    train_y = train_set['isFraud']
    test_X = test_set.drop('isFraud', axis=1)
    test_y = test_set['isFraud']
    print("Test-train Split Complete!")
    return train_X, train_y, test_X, test_y

# Handle Class Imbalance
def balance_class(train_X: pd.DataFrame, train_y: pd.DataFrame):
    print("\nStarting Class Balancing...")

    # Undersampling majority class
    original_minority_count = train_y.value_counts().get(1, 0)
    original_majority_count = train_y.value_counts().get(0, 0)
    target_majority_count  = int(original_majority_count * 0.75)
    if target_majority_count  < original_minority_count and original_majority_count > 0:
        target_majority_count = original_minority_count
    undersampler = RandomUnderSampler(sampling_strategy={0: target_majority_count, 1: original_minority_count}, random_state=42)

    # Oversampling minority class
    oversampler = SMOTE(sampling_strategy=0.1, random_state=42)

    # Pipeline for under and oversampling
    pipeline = Pipeline([('undersample', undersampler),
                         ('oversample', oversampler)])
    train_X_resampled, train_y_resampled = pipeline.fit_resample(train_X, train_y)

    print("Class Balancing Completed!")
    return train_X_resampled, train_y_resampled

def model_training():
    pass

def evaluate_model():
    pass

def save_model():
    pass

if __name__ == "__main__":
    # Load data
    data = load_data()

    # Split Data
    X_train, y_train, X_test, y_test = split_data(data)

    # Balance Data
    X_train_resampled, y_train_resampled = balance_class(X_train, y_train)

    # Train Model
