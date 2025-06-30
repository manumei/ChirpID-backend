''' Las splitting functions para train-val o dev-test '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from utils.data_preparation import create_metadata_dataframe

def try_split_with_seed(df, test_size, seed, min_test_segments, target_test_segments):
    """
    Try a single split with a given seed and evaluate its quality.
    
    Parameters:
    - df: DataFrame with 'class_id', 'author', and 'usable_segments' columns
    - test_size: Target test set size (0.0 to 1.0)
    - seed: Random seed for the split
    - min_test_segments: Minimum total segments per class in test set
    - target_test_segments: Target segments per class for test set
    - total_segments_per_class: Total segments per class in the dataset
    
    Returns:
    - (dev_df, test_df, score) if split is valid, None otherwise
    """
    try:
        gss = GroupShuffleSplit(
            n_splits=1, 
            test_size=test_size,
            random_state=seed
        )

        dev_indices, test_indices = next(gss.split(
            X=df, 
            y=df['class_id'], 
            groups=df['author']
        ))

        dev_df = df.iloc[dev_indices]
        test_df = df.iloc[test_indices]
        
        # Check if all classes are in both sets
        dev_classes = set(dev_df['class_id'])
        test_classes = set(test_df['class_id'])
        all_classes = set(df['class_id'])
        
        if not (all_classes <= dev_classes and all_classes <= test_classes):
            return None  # Skip if missing classes in either set
        
        # Check that each class in dev has at least 2 authors
        dev_authors_per_class = dev_df.groupby('class_id')['author'].nunique()
        if dev_authors_per_class.min() < 2:
            return None  # Skip if any class has fewer than 2 authors in dev
        
        # Check minimum test segments per class
        test_segments_per_class = test_df.groupby('class_id')['usable_segments'].sum()
        if test_segments_per_class.min() < min_test_segments:
            return None  # Skip if any class has too few test segments
        
        # Calculate stratification score based on segments (lower is better)
        actual_test_segments = test_df.groupby('class_id')['usable_segments'].sum().sort_index()
        
        # Mean absolute percentage error from target
        score = np.mean(np.abs(actual_test_segments.values - target_test_segments.values) / target_test_segments.values)
        
        return dev_df.copy(), test_df.copy(), score
    
    except Exception as e:
        raise ValueError(f"Error during split with seed {seed}: {e}") from e
        return None

def search_best_group_seed(df, test_size, max_attempts, min_test_segments):
    """
    Search for the best stratified split while maintaining author grouping.
    Based on total usable segments per class rather than sample counts.
    
    Parameters:
    - df: DataFrame with 'class_id', 'author', and 'usable_segments' columns
    - test_size: Target test set size (0.0 to 1.0)
    - max_attempts: Maximum number of random seeds to try
    - min_test_segments: Minimum total segments per class in test set
    
    Returns:
    - best_dev_df, best_test_df: Best split found
    - best_score: Stratification quality score
    """
    
    # Calculate target distribution based on total segments per class
    total_segments_per_class = df.groupby('class_id')['usable_segments'].sum().sort_index()
    target_test_segments = (total_segments_per_class * test_size).round().astype(int)
    
    best_score = float('inf')
    best_dev_df = None
    best_test_df = None
    best_seed = None
    
    for seed in range(max_attempts):
        result = try_split_with_seed(df, test_size, seed, min_test_segments, target_test_segments)
        
        if result is not None:
            dev_df, test_df, score = result
            
            if score < best_score:
                best_score = score
                best_dev_df = dev_df
                best_test_df = test_df
                best_seed = seed
                
    if best_dev_df is None:
        if min_test_segments < 8:
            raise ValueError("No valid split found with current constraints. Consider relaxing min_test_segments.")
        return search_best_group_seed(df, test_size, max_attempts, min_test_segments=8)
    
    return best_dev_df, best_test_df, best_score

def try_kfold_split_with_seed(df, n_splits, seed, min_val_segments, target_val_segments):
    """
    Try a single K-fold split with a given seed and evaluate its quality.
    
    Parameters:
    - df: DataFrame with 'class_id', 'author', and 'usable_segments' columns
    - n_splits: Number of folds for cross-validation
    - seed: Random seed for the split
    - min_val_segments: Minimum total segments per class in each validation fold
    - target_val_segments: Target segments per class for validation folds
    
    Returns:
    - (folds, avg_score) if split is valid, None otherwise
    """
    try:
        # Try StratifiedGroupKFold first (better for stratification)
        try:
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(skf.split(df, df['class_id'], df['author']))
        except:
            # Fall back to GroupKFold if StratifiedGroupKFold fails
            gkf = GroupKFold(n_splits=n_splits)
            # Shuffle the dataframe for randomness
            df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            splits = list(gkf.split(df_shuffled, df_shuffled['class_id'], df_shuffled['author']))
        
        folds = []
        fold_scores = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            if 'df_shuffled' in locals():
                train_df = df_shuffled.iloc[train_indices]
                val_df = df_shuffled.iloc[val_indices]
            else:
                train_df = df.iloc[train_indices]
                val_df = df.iloc[val_indices]
                # Check if all classes are in training set (validation can have missing classes)
            train_classes = set(train_df['class_id'])
            val_classes = set(val_df['class_id'])
            all_classes = set(df['class_id'])
            
            if not (all_classes <= train_classes):
                return None  # Skip if missing classes in training set
                # Check minimum validation segments per class (only for classes present in validation)
            val_segments_per_class = val_df.groupby('class_id')['usable_segments'].sum()
            if len(val_segments_per_class) > 0 and val_segments_per_class.min() < min_val_segments:
                return None  # Skip if any present class has too few validation segments
                # Calculate stratification score for this fold (only for classes present in validation)
            actual_val_segments = val_df.groupby('class_id')['usable_segments'].sum().sort_index()
            
            # Only compare classes that are actually present in validation set
            if len(actual_val_segments) > 0:
                target_val_segments_present = target_val_segments.loc[actual_val_segments.index]
                fold_score = np.mean(np.abs(actual_val_segments.values - target_val_segments_present.values) / 
                                    np.maximum(target_val_segments_present.values, 1))  # Avoid division by zero
            else:
                fold_score = 0  # No classes in validation set
            
            folds.append((train_df.copy(), val_df.copy()))
            fold_scores.append(fold_score)
        
        # Average score across all folds
        avg_score = np.mean(fold_scores)
        
        return folds, avg_score
    
    except Exception as e:
        return None

def search_best_group_seed_kfold(df, max_attempts, min_val_segments, n_splits):
    """
    Search for the best stratified K-fold split while maintaining author grouping.
    Based on total usable segments per class.
    
    Parameters:
    - df: DataFrame with 'class_id', 'author', and 'usable_segments' columns
    - n_splits: Number of folds for cross-validation
    - max_attempts: Maximum number of random seeds to try
    - min_val_segments: Minimum total segments per class in each validation fold
    
    Returns:
    - best_folds: List of (train_df, val_df) tuples for each fold
    - best_score: Average stratification quality score across all folds
    - best_seed: Random seed that produced the best split
    """
    
    # Calculate target distribution for validation (1/n_splits of total)
    total_segments_per_class = df.groupby('class_id')['usable_segments'].sum().sort_index()
    target_val_segments = (total_segments_per_class / n_splits).round().astype(int)
    
    best_score = float('inf')
    best_folds = None
    best_seed = None
    
    for seed in range(max_attempts):
        if seed % 1200 == 0:
            print(f"Attempt {seed}/{max_attempts - 1}...")
        result = try_kfold_split_with_seed(df, n_splits, seed, min_val_segments, target_val_segments)
        
        if result is not None:
            folds, avg_score = result
            
            if avg_score < best_score:
                best_score = avg_score
                best_folds = folds
                best_seed = seed
    
    if best_folds is None:
        if min_val_segments <= 4:
            raise ValueError("No valid split found with current constraints. Consider relaxing min_val_segments.")
        print("Warning: No valid split found with current constraints. Relaxing min_val_segments...")
        return search_best_group_seed_kfold(df, max_attempts, min_val_segments=5, n_splits=n_splits)
    
    # Print fold statistics
    for i, (train_df, val_df) in enumerate(best_folds):
        author_overlap = set(train_df['author']) & set(val_df['author'])
        if author_overlap:
            raise ValueError("AUTHOR OVERLAP!!")

    if n_splits == 4 and best_folds is not None:
        plot_fold_splits(best_folds)

    return best_folds, best_score, best_seed

def plot_fold_splits(folds):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for fold_idx, (train_df, val_df) in enumerate(folds):
        ax = axes[fold_idx // 2, fold_idx % 2]
        train_counts = train_df['class_id'].value_counts().sort_index()
        val_counts = val_df['class_id'].value_counts().sort_index()
        all_classes = sorted(set(train_counts.index) | set(val_counts.index))
        train_y = [train_counts.get(cls, 0) for cls in all_classes]
        val_y = [val_counts.get(cls, 0) for cls in all_classes]
        ax.plot(all_classes, train_y, label="Train", color="tab:blue")
        ax.plot(all_classes, val_y, label="Val", color="tab:orange")
        ax.set_title(f"Fold {fold_idx+1}")
        ax.set_xlabel("class_id")
        ax.set_ylabel("Num samples")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

# Helper functions for pre-computing splits to optimize configuration sweeping

def precompute_single_fold_split(features, labels, authors, test_size=0.2, 
                                max_attempts=10000, min_test_segments=5):
    """
    Pre-compute a single fold split for use across multiple training configurations.
    This avoids recomputing the same split for different model configurations.
    
    Parameters:
    - features: Feature array (used for creating metadata)
    - labels: Label array
    - authors: Author array
    - test_size: Target test set size (0.0 to 1.0)
    - max_attempts: Maximum number of random seeds to try
    - min_test_segments: Minimum total segments per class in test set
    
    Returns:
    - (train_indices, val_indices, best_score): Pre-computed split indices and score
    """
    
    # Create metadata for splitting
    metadata_df = create_metadata_dataframe(labels, authors)
    
    # Find optimal split with author grouping
    dev_df, test_df, best_score = search_best_group_seed(
        df=metadata_df,
        test_size=test_size,
        max_attempts=max_attempts,
        min_test_segments=min_test_segments
    )
    
    # Extract indices
    train_indices = dev_df['sample_idx'].values
    val_indices = test_df['sample_idx'].values
    
    return train_indices, val_indices, best_score

def precompute_kfold_splits(features, labels, authors, n_splits=4, 
                            max_attempts=30000, min_val_segments=0):
    """
    Pre-compute K-fold splits for use across multiple training configurations.
    This avoids recomputing the same splits for different model configurations.
    
    Parameters:
    - features: Feature array (used for creating metadata)
    - labels: Label array
    - authors: Author array
    - n_splits: Number of folds for cross-validation
    - max_attempts: Maximum number of random seeds to try
    - min_val_segments: Minimum total segments per class in each validation fold
    
    Returns:
    - (fold_indices, best_score, best_seed): Pre-computed fold indices, score, and seed
    """
    
    # Create metadata for splitting
    metadata_df = create_metadata_dataframe(labels, authors)
    
    # Find optimal k-fold splits with author grouping
    best_folds, best_score, best_seed = search_best_group_seed_kfold(
        df=metadata_df,
        max_attempts=max_attempts,
        min_val_segments=min_val_segments,
        n_splits=n_splits
    )
    
    # Convert fold indices for dataset
    fold_indices = []
    for train_df, val_df in best_folds:
        train_indices = train_df['sample_idx'].values
        val_indices = val_df['sample_idx'].values
        fold_indices.append((train_indices, val_indices))
    print(f"   Best seed: {best_seed}")
    
    return fold_indices, best_score, best_seed

def get_set_seed_indices(features, labels, authors, test_size, seed):
    ''' Returns the train_indices and val_indices for a set seed '''
    metadata_df = create_metadata_dataframe(labels, authors)
    dev_df, test_df, best_score = try_split_with_seed(
        df=metadata_df,
        test_size=test_size,
        seed=seed,
        min_test_segments=5,
        target_test_segments=(metadata_df.groupby('class_id')['usable_segments'].sum() * test_size).round().astype(int)
    )
    if dev_df is None or test_df is None:
        raise ValueError(f"Failed to find a valid split with seed {seed}. Try a different seed or adjust parameters.")
    train_indices = dev_df['sample_idx'].values
    val_indices = test_df['sample_idx'].values
    
    return train_indices, val_indices, best_score

def get_set_seed_kfold_indices(features, labels, authors, n_splits, seed):
    ''' Returns the train_indices and val_indices for a set seed in k-fold '''
    metadata_df = create_metadata_dataframe(labels, authors)
    folds, best_score, best_seed = try_kfold_split_with_seed(
        df=metadata_df,
        n_splits=n_splits,
        seed=seed,
        min_val_segments=0,
        target_val_segments=(metadata_df.groupby('class_id')['usable_segments'].sum() / n_splits).round().astype(int)
    )
    
    if folds is None:
        raise ValueError(f"Failed to find a valid k-fold split with seed {seed}. Try a different seed or adjust parameters.")
    
    fold_indices = []
    for train_df, val_df in folds:
        train_indices = train_df['sample_idx'].values
        val_indices = val_df['sample_idx'].values
        fold_indices.append((train_indices, val_indices))
    
    return fold_indices, best_score, best_seed

def display_split_statistics(split_data, split_type="single"):
    """
    Display statistics about pre-computed splits for verification.
    
    Parameters:
    - split_data: Either (train_indices, val_indices, score) for single fold
                  or (fold_indices, score, seed) for k-fold
    - split_type: "single" or "kfold"
    """
    print(f"\n📊 {split_type.upper()} SPLIT STATISTICS")
    print("-" * 40)
    
    if split_type == "single":
        train_indices, val_indices, score = split_data
        print(f"Train samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Split ratio: {len(train_indices)/(len(train_indices)+len(val_indices)):.2%} - {len(val_indices)/(len(train_indices)+len(val_indices)):.2%}")
        print(f"Quality score: {score:.4f}")
        
    elif split_type == "kfold":
        fold_indices, score, seed = split_data
        print(f"Number of folds: {len(fold_indices)}")
        print(f"Random seed: {seed}")
        print(f"Average quality score: {score:.4f}")
        
        for i, (train_idx, val_idx) in enumerate(fold_indices):
            print(f"  Fold {i+1}: {len(train_idx)} train, {len(val_idx)} val ({len(train_idx)/(len(train_idx)+len(val_idx)):.2%} - {len(val_idx)/(len(train_idx)+len(val_idx)):.2%})")
    
    print("-" * 40)