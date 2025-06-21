import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

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
        return None

def group_split_with_stratification_search(df, test_size, max_attempts, min_test_segments):
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
                
                print(f"New best split found! Seed: {seed}, Score: {score:.3f}")
    
    if best_dev_df is None:
        if min_test_segments < 10:
            raise ValueError("No valid split found with current constraints. Consider relaxing min_test_segments.")
        return group_split_with_stratification_search(df, test_size, max_attempts, min_test_segments=10)
    
    print(f"\nBest split found:")
    print(f"Seed: {best_seed}")
    print(f"Stratification score: {best_score:.3f}")
    print(f"Author overlap: {set(best_dev_df['author']) & set(best_test_df['author'])}")

    print(f"Segments in dev set: {best_dev_df['usable_segments'].sum()}")
    print(f"Segments in test set: {best_test_df['usable_segments'].sum()}")
    print(f"Dev segment%: {best_dev_df['usable_segments'].sum() / df['usable_segments'].sum():.2%}")
    print(f"Test segment%: {best_test_df['usable_segments'].sum() / df['usable_segments'].sum():.2%}")

    # Print distribution comparison
    print("\nSegment distribution comparison:")
    actual_test_segments = best_test_df.groupby('class_id')['usable_segments'].sum().sort_index()
    dev_segments = best_dev_df.groupby('class_id')['usable_segments'].sum().sort_index()
    target_dev_segments = total_segments_per_class - target_test_segments
    
    comparison_df = pd.DataFrame({
        'Target_Test_Segments': target_test_segments,
        'Actual_Test_Segments': actual_test_segments,
        'Target_Dev_Segments': target_dev_segments,
        'Actual_Dev_Segments': dev_segments,
        'Total_Segments': total_segments_per_class
    })
    
    print(tabulate(comparison_df, headers=comparison_df.columns, tablefmt='grid'))
    
    return best_dev_df, best_test_df, best_score

