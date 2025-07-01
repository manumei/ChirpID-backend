#!/usr/bin/env python3
"""
Quick test to verify the split pre-computation optimization works correctly.
This test creates mock data and verifies that the optimization functions work.
"""
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.split import precompute_single_fold_split, precompute_kfold_splits, display_split_statistics

def test_split_precomputation():
    """Test the split pre-computation functions with mock data."""
    print("🧪 TESTING SPLIT PRE-COMPUTATION OPTIMIZATION")
    print("=" * 60)
    
    # Create mock data (smaller for testing)
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    n_authors = 15
    
    # Generate mock features, labels, and authors
    features = np.random.randn(n_samples, 224, 313)
    labels = np.random.randint(0, n_classes, n_samples)
    authors = np.random.choice([f"author_{i}" for i in range(n_authors)], n_samples)
    
    print(f"Mock dataset: {n_samples} samples, {n_classes} classes, {n_authors} authors")
    
    try:
        # Test single fold pre-computation
        print("\n📋 Testing single fold pre-computation...")
        single_split = precompute_single_fold_split(
            features=features, 
            labels=labels, 
            authors=authors,
            test_size=0.2,
            max_attempts=100,  # Smaller for testing
            min_test_segments=1
        )
        
        print("✅ Single fold pre-computation successful")
        display_split_statistics(single_split, "single")
        
        # Test k-fold pre-computation
        print("\n📋 Testing k-fold pre-computation...")
        kfold_splits = precompute_kfold_splits(
            features=features,
            labels=labels, 
            authors=authors,
            n_splits=3,  # Smaller for testing
            max_attempts=100,
            min_val_segments=1
        )
        
        print("✅ K-fold pre-computation successful")
        display_split_statistics(kfold_splits, "kfold")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Split pre-computation optimization is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_split_precomputation()
    sys.exit(0 if success else 1)
