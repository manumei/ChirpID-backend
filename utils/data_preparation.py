# Data Preparation Utilities
# Handles data loading and preprocessing for training

import os
import numpy as np
import pandas as pd


def prepare_training_data(data_path=None, features=None, labels=None, authors=None):
    """
    Load and prepare training data from file or arrays.
    
    Args:
        data_path (str, optional): Path to CSV file with training data
        features (np.ndarray, optional): Pre-loaded feature array
        labels (np.ndarray, optional): Pre-loaded labels array
        authors (np.ndarray, optional): Pre-loaded authors array
    
    Returns:
        tuple: (features, labels, authors) as numpy arrays
    """
    if data_path is not None:
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Extract components
        labels = df['label'].values.astype(np.int64)
        authors = df['author'].values
        features = df.drop(columns=['label', 'author']).values.astype(np.float32)
        
        # Normalize features to 0-1 range
        features /= 255.0
        
        # Reshape for CNN input (batch_size, channels, height, width)
        features = features.reshape(-1, 1, 313, 224)
        
        # print(f"Loaded data from CSV:")
        # print(f"  Features shape: {features.shape}")
        # print(f"  Labels shape: {labels.shape}")
        # print(f"  Authors shape: {authors.shape}")
        
    elif features is not None and labels is not None and authors is not None:
        print("Using provided arrays")
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray) or not isinstance(authors, np.ndarray):
            raise TypeError("Features, labels, and authors must be numpy arrays")
        
    else:
        raise ValueError("Either data_path or (features, labels, authors) must be provided")
    
    # Validation
    if len(features) != len(labels) or len(labels) != len(authors):
        raise ValueError("Features, labels, and authors must have the same length")
    
    return features, labels, authors


def create_metadata_dataframe(labels, authors, sample_indices=None):
    """
    Create metadata DataFrame for splitting operations.
    
    Args:
        labels (np.ndarray): Array of class labels
        authors (np.ndarray): Array of author identifiers
        sample_indices (np.ndarray, optional): Array of sample indices
    
    Returns:
        pd.DataFrame: Metadata DataFrame with columns for splitting
    """
    if sample_indices is None:
        sample_indices = np.arange(len(labels))
    
    metadata_df = pd.DataFrame({
        'sample_idx': sample_indices,
        'class_id': labels,
        'author': authors,
        'usable_segments': 1  # Each sample represents 1 segment
    })
    
    # print(f"Created metadata DataFrame:")
    # print(f"  Shape: {metadata_df.shape}")
    # print(f"  Unique authors: {len(metadata_df['author'].unique())}")
    # print(f"  Unique classes: {len(metadata_df['class_id'].unique())}")
    
    return metadata_df


def validate_data_integrity(features, labels, authors):
    """
    Validate data integrity and consistency.
    
    Args:
        features (np.ndarray): Feature array
        labels (np.ndarray): Labels array
        authors (np.ndarray): Authors array
    
    Raises:
        ValueError: If data integrity issues are found
    """
    # Check lengths
    if not (len(features) == len(labels) == len(authors)):
        raise ValueError(f"Length mismatch: features={len(features)}, labels={len(labels)}, authors={len(authors)}")
    
    # Check for NaN values
    if np.any(np.isnan(features)):
        raise ValueError("Features contain NaN values")
    
    if np.any(pd.isna(labels)):
        raise ValueError("Labels contain NaN values")
    
    if np.any(pd.isna(authors)):
        raise ValueError("Authors contain NaN values")
    
    # Check data types
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"Features must be float type, got {features.dtype}")
    
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Labels must be integer type, got {labels.dtype}")
    
    # Check value ranges
    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative")
    
    if features.ndim != 4:
        raise ValueError(f"Features must be 4D (batch, channels, height, width), got shape {features.shape}")
    
    print("Data integrity validation passed ✓")


def get_data_statistics(features, labels, authors):
    """
    Print comprehensive data statistics.
    
    Args:
        features (np.ndarray): Feature array
        labels (np.ndarray): Labels array
        authors (np.ndarray): Authors array
    """
    print("\\n" + "=" * 50)
    print("DATA STATISTICS")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total samples: {len(features)}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Feature mean: {features.mean():.4f}")
    print(f"Feature std: {features.std():.4f}")
    
    # Class distribution
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"\\nClass distribution:")
    print(f"  Number of classes: {len(unique_labels)}")
    print(f"  Class range: [{unique_labels.min()}, {unique_labels.max()}]")
    print(f"  Samples per class: min={label_counts.min()}, max={label_counts.max()}, mean={label_counts.mean():.1f}")
    
    # Author distribution
    unique_authors, author_counts = np.unique(authors, return_counts=True)
    print(f"\\nAuthor distribution:")
    print(f"  Number of authors: {len(unique_authors)}")
    print(f"  Samples per author: min={author_counts.min()}, max={author_counts.max()}, mean={author_counts.mean():.1f}")
    
    # Check for potential issues
    if label_counts.min() < 10:
        print(f"\\n⚠️  Warning: Some classes have very few samples (min: {label_counts.min()})")
    
    if author_counts.min() < 2:
        print(f"\\n⚠️  Warning: Some authors have very few samples (min: {author_counts.min()})")
    
    print("=" * 50)
