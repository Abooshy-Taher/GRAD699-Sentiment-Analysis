"""
Utility functions for Week 5: Unsloth Fine-tuning Pipeline
Reusable functions extracted from Week 4 notebook for data loading, cleaning, and splitting.
"""

import pandas as pd
import numpy as np
import os


def load_dataset():
    """
    Load Amazon_Data.csv with support for local and Colab environments.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    possible_paths = [
        "/Users/abdullah/Desktop/HU Classes/GRAD699/Sentiment Analysis/Amazon_Data.csv",
        "../Amazon_Data.csv",
        "../../Amazon_Data.csv",
        "Amazon_Data.csv",
    ]
    
    # Check if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
        possible_paths.extend([
            "/content/drive/MyDrive/Amazon_Data.csv",
            "/content/Amazon_Data.csv",
        ])
    except:
        IN_COLAB = False
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            csv_path = path
            print(f"✓ Found file at: {path}")
            break
    
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find Amazon_Data.csv in any of the expected locations.\n"
            f"If running in Colab, please upload Amazon_Data.csv to /content/drive/MyDrive/ or /content/"
        )
    
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    return df


def clean_data(df):
    """
    Clean the dataset: remove nulls, convert timestamp, remove empty text.
    
    Args:
        df: Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Keep only necessary columns: text, rating, timestamp
    df = df[['text', 'rating', 'timestamp']].copy()
    
    # Remove rows with missing values
    print(f"Before cleaning: {len(df):,} rows")
    df = df.dropna()
    print(f"After removing nulls: {len(df):,} rows")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    print(f"After timestamp conversion: {len(df):,} rows")
    
    # Remove empty text reviews
    df = df[df['text'].astype(str).str.len() > 0].copy()
    print(f"After removing empty text: {len(df):,} rows")
    
    # Sort by timestamp (critical for chronological splitting)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def rating_to_sentiment_label(rating):
    """
    Convert rating to sentiment label.
    
    Args:
        rating: Integer rating (1-5)
        
    Returns:
        int: Sentiment label (0=Negative, 1=Neutral, 2=Positive)
    """
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:  # rating >= 4
        return 2  # Positive


def create_sentiment_labels(df):
    """
    Create sentiment labels from ratings.
    
    Args:
        df: Dataframe with 'rating' column
        
    Returns:
        pd.DataFrame: Dataframe with 'sentiment_label' column added
    """
    df = df.copy()
    df['sentiment_label'] = df['rating'].apply(rating_to_sentiment_label)
    
    label_counts = df['sentiment_label'].value_counts().sort_index()
    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION (Rating-Based Labels)")
    print("=" * 60)
    print(f"  Negative (1-2 stars): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Neutral (3 stars):     {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Positive (4-5 stars):  {label_counts.get(2, 0):,} ({label_counts.get(2, 0)/len(df)*100:.1f}%)")
    print("=" * 60)
    
    return df


def chronological_split(df, train_ratio=0.70, val_ratio=0.15):
    """
    Split dataframe chronologically by timestamp.
    
    Args:
        df: Dataframe sorted by timestamp
        train_ratio: Proportion for training set (default 0.70)
        val_ratio: Proportion for validation set (default 0.15)
        
    Returns:
        tuple: (df_train, df_val, df_test, y_train, y_val, y_test)
    """
    n_total = len(df)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Chronological splits
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()
    
    # Extract targets
    y_train = df_train['sentiment_label'].values
    y_val = df_val['sentiment_label'].values
    y_test = df_test['sentiment_label'].values
    
    print("=" * 60)
    print("CHRONOLOGICAL SPLIT COMPLETE")
    print("=" * 60)
    print(f"Train set: {len(df_train):,} samples ({len(df_train)/n_total*100:.1f}%)")
    print(f"  Date range: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"\nValidation set: {len(df_val):,} samples ({len(df_val)/n_total*100:.1f}%)")
    print(f"  Date range: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"\nTest set: {len(df_test):,} samples ({len(df_test)/n_total*100:.1f}%)")
    print(f"  Date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    
    # Sanity check: no overlap in dates
    assert df_train['timestamp'].max() <= df_val['timestamp'].min(), "Train/Val overlap!"
    assert df_val['timestamp'].max() <= df_test['timestamp'].min(), "Val/Test overlap!"
    print("\n✓ Sanity checks passed: no temporal overlap")
    print("=" * 60)
    
    return df_train, df_val, df_test, y_train, y_val, y_test


def label_to_string(label):
    """
    Convert numeric label to string.
    
    Args:
        label: Integer label (0, 1, or 2)
        
    Returns:
        str: "Negative", "Neutral", or "Positive"
    """
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[label]


def string_to_label(label_str):
    """
    Convert string label to numeric.
    
    Args:
        label_str: String label ("Negative", "Neutral", or "Positive")
        
    Returns:
        int: Numeric label (0, 1, or 2)
    """
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    # Handle case-insensitive matching and partial matches
    label_str_lower = label_str.strip().lower()
    for key, value in label_map.items():
        if key.lower() in label_str_lower or label_str_lower in key.lower():
            return value
    # Default fallback
    return 1  # Neutral if unclear

