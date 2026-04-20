import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import NUM_FEATURE, PACKET_NUM, SEED, TARGET_TRAIN_LABELED_RATIO

def data_processing(df):
    """
    Processes the input dataframe for traffic classification.
    
    Args:
        df (pd.DataFrame): Input dataframe containing flow data and Label.
        
    Returns:
        tuple: (X, y) where X is the reshaped feature matrix and y is the labels.
    """
    # Extract labels
    y = df['Label'].to_numpy()
    
    # Remove non-feature columns
    # Assuming 'flow_id' and 'Label' are the columns to drop
    X_df = df.drop(['Label', 'flow_id'], axis=1)
    
    # Normalize features to [0, 1]
    X = X_df.to_numpy() / 255.0
    
    # Reshape to (samples, packet_num, num_features)
    X = X.reshape(-1, PACKET_NUM, NUM_FEATURE)
    
    # Handle the labels for flows (take the label of the last packet in the flow)
    # The original code did: y_train = y_train.reshape(-1,20)[:,-1]
    y = y.reshape(-1, PACKET_NUM)[:, -1]
    
    return X.astype(np.float32), y.astype(np.int64)

def load_source_data(path):
    """Loads source domain data (full dataset for training)."""
    try:
        df = pd.read_feather(path)
        return data_processing(df)
    except Exception as e:
        print(f"Error loading source data from {path}: {e}")
        return None, None

def load_target_data(path, test_ratio=0.2, seed=None):
    """
    Loads target domain data and splits into labeled, unlabeled, and test sets.
    
    Args:
        path: Path to target domain feather file.
        test_ratio: Ratio of data for testing (default 0.2 = 20%).
        seed: Random seed for reproducibility. Defaults to config SEED.
        
    Returns:
        tuple: (labeled_X, labeled_y, unlabeled_X, unlabeled_y, test_X, test_y)
    """
    if seed is None:
        seed = SEED
    try:
        df = pd.read_feather(path)
        X, y = data_processing(df)
        
        # Step 1: Split target domain into Semi-supervised (train) and Test
        X_semi, X_test, y_semi, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=seed, stratify=y
        )
        
        # Step 2: Split Semi-supervised into Labeled and Unlabeled
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X_semi, y_semi, train_size=TARGET_TRAIN_LABELED_RATIO, random_state=seed, stratify=y_semi
        )
        
        print(f"Target domain: {len(X)} samples")
        print(f"  -> Test: {len(X_test)}")
        print(f"  -> Semi-supervised: {len(X_semi)} (Labeled: {len(X_labeled)}, Unlabeled: {len(X_unlabeled)})")
        
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test
    except Exception as e:
        print(f"Error loading target data from {path}: {e}")
        return None, None, None, None, None, None
