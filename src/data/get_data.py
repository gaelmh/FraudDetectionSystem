import os
import kaggle
import pandas as pd

# Dataset path on Kaggle
KAGGLE_DATASET = "ealaxi/paysim1"
KAGGLE_FILE_NAME = "PS_20174392719_1491204439457_log.csv"

# Directory where raw data will be stored
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw')
DATA_FILE_PATH = os.path.join(RAW_DATA_DIR, KAGGLE_FILE_NAME)

def download_kaggle_dataset():
    """
    Downloads the specified Kaggle dataset to the raw data directory.
    """
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"Created directory: {RAW_DATA_DIR}")

    print(f"Attempting to download {KAGGLE_FILE_NAME} from Kaggle dataset: {KAGGLE_DATASET}")
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=RAW_DATA_DIR, unzip=True)
        print(f"Successfully downloaded and unzipped data to {RAW_DATA_DIR}")

        # Verify if the specific file exists after unzipping
        if os.path.exists(DATA_FILE_PATH):
            print(f"Found {KAGGLE_FILE_NAME} at {DATA_FILE_PATH}")
        else:
            # Add more descriptive warning for debugging
            print(f"Warning: {KAGGLE_FILE_NAME} not found at {DATA_FILE_PATH} after unzipping.")
            print(f"Please check the actual contents of {RAW_DATA_DIR} to verify file name or subfolder issue.")

    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Please ensure your kaggle.json is correctly placed and contains valid credentials.")

def load_raw_data():
    """
    Loads the raw CSV data into a pandas DataFrame.
    """
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Data file not found at {DATA_FILE_PATH}. Please ensure it's downloaded and named correctly.")
        return None

    print(f"Loading raw data from {DATA_FILE_PATH}...")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Raw data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading CSV file from {DATA_FILE_PATH}: {e}")
        return None

if __name__ == "__main__":
    download_kaggle_dataset()

    # Loading to confirm everything works
    df_raw = load_raw_data()
    if df_raw is not None:
        print("\nFirst 5 rows of the loaded data:")
        print(df_raw.head())
        print("\nColumns in the loaded data:")
        print(df_raw.columns.tolist())