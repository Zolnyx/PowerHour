import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_validate():
    """Combined loading with strict validation"""
    # Load inputs
    df_in = pd.read_csv("data/input_vectors.csv", header=None)
    X = df_in.iloc[:, 2:7].values.astype(np.float32)
    
    # Load outputs
    df_out = pd.read_csv("data/output_vectors.csv", header=None)
    y = df_out.iloc[:, 2:8].values.astype(np.float32)
    
    # Synchronization check
    assert len(X) == len(y), "Input/output length mismatch"
    assert (df_in[0] == df_out[0]).all(), "Video ID mismatch"
    assert (df_in[1] == df_out[1]).all(), "Frame number mismatch"
    
    # Filter invalid
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    return X[valid_mask], y[valid_mask]

def get_train_test():
    X, y = load_and_validate()
    return train_test_split(X, y, test_size=0.2, random_state=42)