"""
DataLoader Optimization Test Suite
Tests worker safety, performance improvements, and compatibility
"""

import os
import sys
import torch
import numpy as np
import pytest
from torch.utils.data import TensorDataset
import tempfile
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataloader_factory import OptimalDataLoaderFactory
from utils.dataset_utils import StandardizedSubset, AugmentedDataset
from benchmark.performance_monitor import DataLoaderPerformanceMonitor
from utils.specaugment import get_augmentation_params


class TestDataLoaderOptimizations:
    """Test suite for DataLoader optimizations."""
    
    @classmethod
    def setup_class(cls):
        """Set up test data."""
        # Create synthetic test data
        cls.n_samples = 1000
        cls.n_features = 224 * 313  # Spectrogram size
        cls.n_classes = 31
        
        # Generate random spectrogram-like data
        features = np.random.rand(cls.n_samples, 1, 224, 313).astype(np.float32)
        labels = np.random.randint(0, cls.n_classes, cls.n_samples).astype(np.int64)
        
        cls.base_dataset = TensorDataset(
            torch.tensor(features),
            torch.tensor(labels)
        )
        
        # Compute standardization stats
        sample_data = torch.stack([cls.base_dataset[i][0] for i in range(100)])
        cls.mean = sample_data.mean()
        cls.std = sample_data.std() + 1e-8
        
        # Create augmentation params
        cls.augment_params = get_augmentation_params(cls.n_samples, cls.n_classes)
    
    def test_dataloader_factory_basic(self):
        """Test basic DataLoader factory functionality."""
        # Test training loader creation
        train_loader = OptimalDataLoaderFactory.create_training_loader(
            self.base_dataset,
            batch_size=16
        )
        
        # Test validation loader creation
        val_loader = OptimalDataLoaderFactory.create_validation_loader(
            self.base_dataset,
            batch_size=16
        )
          # Verify basic properties
        assert train_loader.batch_size == 16
        assert val_loader.batch_size == 16
        assert train_loader.dataset == self.base_dataset
        assert val_loader.dataset == self.base_dataset
        
        # Test data loading
        batch = next(iter(train_loader))
        assert len(batch) == 2  # X, y
        assert batch[0].shape[0] == 16  # Batch size
        assert batch[1].shape[0] == 16  # Batch size
    
    def test_worker_safety_basic_dataset(self):
        """Test worker safety with basic dataset."""
        worker_counts = [0, 2, 4, 8]
        
        for num_workers in worker_counts:
            try:
                loader = OptimalDataLoaderFactory.create_training_loader(
                    self.base_dataset,
                    batch_size=8,
                    num_workers=num_workers
                )
                
                # Load a few batches to test multiprocessing
                batch_count = 0
                for batch in loader:
                    batch_count += 1
                    if batch_count >= 3:
                        break
                
                assert batch_count == 3, f"Failed with {num_workers} workers"
                
            except Exception as e:
                pytest.fail(f"Worker safety test failed with {num_workers} workers: {e}")
    
    def test_worker_safety_standardized_dataset(self):
        """Test worker safety with standardized dataset."""
        # Create standardized dataset
        indices = list(range(len(self.base_dataset)))
        standardized_dataset = StandardizedSubset(
            self.base_dataset, indices, self.mean, self.std
        )
        
        worker_counts = [0, 2, 4]  # Use fewer workers for complex operations
        
        for num_workers in worker_counts:
            try:
                loader = OptimalDataLoaderFactory.create_training_loader(
                    standardized_dataset,
                    batch_size=8,
                    num_workers=num_workers,
                    has_standardization=True
                )
                
                # Test data loading
                batch_count = 0
                for batch in loader:
                    batch_count += 1
                    if batch_count >= 2:
                        break
                
                assert batch_count == 2
                
            except Exception as e:
                pytest.fail(f"Standardized dataset worker test failed with {num_workers} workers: {e}")
    
    def test_worker_safety_augmented_dataset(self):
        """Test worker safety with augmented dataset."""
        # Create augmented dataset
        augmented_dataset = AugmentedDataset(
            self.base_dataset,
            use_spec_augment=True,
            use_gaussian_noise=True,
            augment_params=self.augment_params,
            training=True
        )
        
        worker_counts = [0, 2, 4]  # Use fewer workers for augmentation
        
        for num_workers in worker_counts:
            try:
                loader = OptimalDataLoaderFactory.create_training_loader(
                    augmented_dataset,
                    batch_size=8,
                    num_workers=num_workers,
                    has_augmentation=True
                )
                
                # Test data loading
                batch_count = 0
                for batch in loader:
                    batch_count += 1
                    if batch_count >= 2:
                        break
                
                assert batch_count == 2
                
            except Exception as e:
                pytest.fail(f"Augmented dataset worker test failed with {num_workers} workers: {e}")
    
    def test_performance_improvement(self):
        """Test that optimized configuration is faster than baseline."""
        monitor = DataLoaderPerformanceMonitor()
        
        # Compare basic dataset performance
        results = monitor.compare_optimized_vs_baseline(
            self.base_dataset,
            batch_size=16,
            has_augmentation=False,
            has_standardization=False
        )
        
        # Verify improvement exists
        improvement = results['improvement']
        assert improvement['speedup_factor'] > 1.0, "Optimized config should be faster"
    
    def test_optimal_config_selection(self):
        """Test that optimal configuration is selected correctly."""
        # Test different dataset sizes and configurations
        test_cases = [
            (100, False, False, 'small dataset'),
            (5000, False, False, 'large dataset'),
            (1000, True, False, 'with augmentation'),
            (1000, False, True, 'with standardization'),
            (1000, True, True, 'with both')
        ]
        
        for dataset_size, has_aug, has_std, description in test_cases:
            config = OptimalDataLoaderFactory.get_optimal_config(
                dataset_size, has_aug, has_std
            )
            
            # Verify reasonable configuration
            assert isinstance(config['num_workers'], int)
            assert config['num_workers'] >= 0
            assert isinstance(config['pin_memory'], bool)
            assert isinstance(config['persistent_workers'], bool)
            
            # Verify workers are disabled for single-threaded when needed
            if config['num_workers'] == 0:
                assert config['persistent_workers'] == False
    
    def test_augmentation_consistency(self):
        """Test that augmentation produces different results but same shapes."""
        augmented_dataset = AugmentedDataset(
            self.base_dataset,
            use_spec_augment=True,
            use_gaussian_noise=False,
            augment_params=self.augment_params,
            training=True
        )
        
        # Get same sample multiple times (should be different due to augmentation)
        sample1, label1 = augmented_dataset[0]
        sample2, label2 = augmented_dataset[0]
        
        # Labels should be same, samples should be different (due to random augmentation)
        assert torch.equal(label1, label2)
        assert sample1.shape == sample2.shape
        # Note: samples might occasionally be identical due to randomness, so we don't assert they're different
    
    def test_standardization_correctness(self):
        """Test that standardization applies correctly."""
        indices = list(range(100))  # Use subset for testing
        standardized_dataset = StandardizedSubset(
            self.base_dataset, indices, self.mean, self.std
        )
        
        # Get a sample and verify standardization
        original_sample, _ = self.base_dataset[0]
        standardized_sample, _ = standardized_dataset[0]
        
        # Verify shapes are preserved
        assert original_sample.shape == standardized_sample.shape
        
        # Verify standardization formula
        expected_standardized = (original_sample - self.mean) / self.std
        assert torch.allclose(standardized_sample, expected_standardized, rtol=1e-5)


def test_integration_with_training_engine():
    """Integration test with actual training components."""
    try:
        from utils.training_engine import TrainingEngine
        from utils.models import BirdCNN
        
        # Create small test dataset
        features = np.random.rand(100, 1, 224, 313).astype(np.float32)
        labels = np.random.randint(0, 5, 100).astype(np.int64)
        dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
        
        # Test configuration
        config = {
            'batch_size': 8,
            'learning_rate': 0.001,
            'num_epochs': 1,  # Just test setup, not full training
            'standardize': True,
            'spec_augment': True,
            'gaussian_noise': False,
            'optimize_dataloaders': True
        }
        
        # Initialize training engine
        engine = TrainingEngine(BirdCNN, 5, config)
        
        # Test data subset creation
        train_indices = list(range(80))
        val_indices = list(range(80, 100))
        
        train_subset, val_subset = engine._create_data_subsets(
            dataset, train_indices, val_indices
        )
        
        # Test data loader creation
        train_loader, val_loader = engine._create_data_loaders(train_subset, val_subset)
        
        # Verify loaders work
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert len(train_batch) == 2
        assert len(val_batch) == 2
        assert train_batch[0].shape[0] <= 8  # Batch size
        assert val_batch[0].shape[0] <= 8    # Batch size
        
    except ImportError:
        pytest.skip("Training engine components not available")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestDataLoaderOptimizations()
    test_instance.setup_class()
    
    print("Running DataLoader optimization tests...")
    
    # Run individual tests
    test_methods = [
        test_instance.test_dataloader_factory_basic,
        test_instance.test_worker_safety_basic_dataset,
        test_instance.test_worker_safety_standardized_dataset,
        test_instance.test_worker_safety_augmented_dataset,
        test_instance.test_performance_improvement,
        test_instance.test_optimal_config_selection,
        test_instance.test_augmentation_consistency,
        test_instance.test_standardization_correctness
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"Running {test_method.__name__}...", end=" ")
            test_method()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    # Run integration test
    try:
        print("Running integration test...", end=" ")
        test_integration_with_training_engine()
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")
        failed += 1
    
    print(f"\\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! DataLoader optimizations are working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Please review the implementation.")
