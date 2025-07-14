import os
import json
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Input data
FILE_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/data/processed/'
FILE_NAME = 'processed_data.parquet'
FULL_PATH = os.path.join(FILE_PATH, FILE_NAME)

# Config file
CONFIG_PATH = 'C:/Users/gaelm/OneDrive - DePaul University/Gael/Trabajo/Fraud Detection System/src/models/'
CONFIG_NAME = 'config.json'
CONFIG_FULL_PATH = os.path.join(CONFIG_PATH, CONFIG_NAME)

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
    target_majority_count  = int(original_majority_count * 0.5)
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

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def model_training(X_train: pd.DataFrame, y_train: pd.Series, config_path: str):
    print("Starting Model Training...")

    # Loading configuration file
    config = load_config(CONFIG_FULL_PATH)
    model_config_section = config['model_training']
    active_model_name = config['active_model']
    selected_model_params = None
    for model_params in model_config_section['models_to_train']:
        if model_params['name'] == active_model_name:
            selected_model_params = model_params
            break
    if selected_model_params is None:
        raise ValueError(f"Active model '{active_model_name}' not found in config.json under 'models_to_train'.")

    model_instance = None
    if active_model_name == 'logistic_regression':
        model_instance = LogisticRegression(**selected_model_params['fixed_params'])
    elif active_model_name == 'random_forest':
        model_instance = RandomForestClassifier(**selected_model_params['fixed_params'])
    # Add more elif conditions for other models (e.g., SVM, GradientBoosting)
    else:
        raise ValueError(f"Unsupported active model: {active_model_name}")

    # GridSearchCV
    param_grid = selected_model_params['grid_search_params']
    scoring_metric = model_config_section['grid_search_scoring']
    cv_strategy = StratifiedKFold(n_splits=model_config_section.get('n_splits_cv', 5))

    grid_search = GridSearchCV(
        estimator=model_instance,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv_strategy,
        n_jobs= 6,
        verbose= 2
    )

    print(f"Training {active_model_name} with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\nModel Training Complete for {active_model_name}!")
    print(f"Best Score ({scoring_metric}): {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

    return best_model, best_params, best_score

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
    best_trained_model, best_params, best_score = model_training(X_train_resampled, y_train_resampled, CONFIG_FULL_PATH)
