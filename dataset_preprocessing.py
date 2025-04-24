import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from constants import DATASET_PATH, RANDOM_SEED
from log import _logger

def load_dataset(ds_path = DATASET_PATH, verbose = True):
    """
    Load parquet data from the specified path.
    
    Args:
        ds_path (str): Path to the dataset directory.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    parquet_files = [
        os.path.join(ds_path, f) for f in os.listdir(ds_path) if f.endswith('.parquet')
    ]

    # Load parquet files into a DataFrame
    dataframes = []
    for path in parquet_files:
        temp_df = pd.read_parquet(path, engine='pyarrow')
        dataframes.append(temp_df)

    # Concatenate all DataFrames into one
    df = pd.concat(dataframes, ignore_index=True)

    if verbose:
        _logger.info("\n*** Raw Dataset Statistics ***\n")
        _logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {len(parquet_files)} files.")
        _logger.info("\nFirst few rows of the DataFrame:")
        _logger.info(df.head())

        # Display basic info
        _logger.info("\nDataset Info:")
        _logger.info(df.info())

        # Display available labels
        labels = df['Label'].unique()
        _logger.info(f"\nAvailable Labels: {labels}")
        _logger.info(f"\nTotal Labels: {len(labels)}")

    return df

def clean_dataset(df, attacks = 'all'):
    """
    Clean the dataset by removing unnecessary columns and renaming others.
    Args:
        attacks (str/list): Type of attack to filter the dataset. Avaiable options are:
            'all' (default): Keep all attacks.
            'benign': Keep only benign entries.
            'DoS': Keep only DoS entries.
            'postscan': Keep only postscan entries.
            'reconnaissance': Keep only reconnaissance entries.
            'exploits': Keep only exploits entries.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Filter attacks if necessary
    if attacks != 'all':
        labels = ['benign']
        if isinstance(attacks, str):
            labels.append(attacks)
        elif isinstance(attacks, list):
            labels.extend(attacks)

        df = df[df['Label'].str.lower().isin(labels)].copy()

    # Reference: https://www.kaggle.com/code/pranavjha24/cic-ids-2017-xgboost-classification
    # Check for missing values
    null_values = df.isnull().sum()
    _logger.info(f"\nMissing Values by Column: \n {null_values}")
    _logger.info(f"\nTotal Null Entries: {null_values.sum()}")

    # Drop na values
    df.dropna(inplace=True)

    duplicate_count = df.duplicated().sum()
    _logger.info(f"Total duplicate entries found: {duplicate_count}\n")
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Reset row index
    df.reset_index(drop=True, inplace=True)

    return df

def extract_features_and_labels(df):
    """
    Extract features from the dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    features = df.drop(columns=['Label'])
    labels = (df['Label'].str.lower()  != 'benign').astype(int) # 1 for attack, 0 for benign
    return features, labels

def balance_dataset(features, labels, ratio=0.25):
    """
    Balance the dataset using RandomUnderSampler.
    Args:
        features (pd.DataFrame): DataFrame containing the features.
        labels (pd.Series): Series containing the labels.
        ratio (float): Desired ratio of benign to attack samples.
            Default is 0.1 (1 attack : 10 benign).
    Returns:
        pd.DataFrame: Balanced features DataFrame.
        pd.Series: Balanced labels Series.
    """
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=RANDOM_SEED)
    features_resampled, labels_resampled = rus.fit_resample(features, labels)
    return features_resampled, labels_resampled 

if __name__ == '__main__':
    # Example usage
    dataset = load_dataset()
    _logger.info(f"Dataset shape: {dataset.shape}")
    cleaned_dataset = clean_dataset(dataset, attacks='all')
    _logger.info(f"Cleaned dataset shape: {cleaned_dataset.shape}")
    features, labels = extract_features_and_labels(cleaned_dataset)
    
    _logger.info(features.info())
    _logger.info(f"Features shape: {features.shape}")
    _logger.info(f"Labels shape: {labels.shape}")
    check_attack_ratio(labels)

    features_balanced, labels_balanced = balance_dataset(features, labels)
    check_attack_ratio(labels_balanced)