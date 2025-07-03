#!/usr/bin/env python3
"""
Test script to verify the new optimizer options work correctly.
Tests SGD, Adam, and AdamW optimizer initialization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from utils.training_engine import TrainingEngine

# Simple test model
class TestModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def test_optimizer_options():
    """Test all three optimizer options"""
    print("Testing optimizer options...")
    
    # Create dummy data
    features = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(features, labels)
    
    # Test configurations
    configs = [
        {'use_adam': False, 'name': 'SGD'},
        {'use_adam': True, 'name': 'Adam (True)'},
        {'use_adam': 'adam', 'name': 'Adam (string)'},
        {'use_adam': 'adamw', 'name': 'AdamW'}
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        # Add required config parameters
        full_config = {
            'batch_size': 16,
            'learning_rate': 0.001,
            'l2_regularization': 1e-4,
            'momentum': 0.9,
            **config
        }
        
        try:
            # Create training engine
            engine = TrainingEngine(TestModel, num_classes=10, config=full_config)
            
            # Test component initialization
            train_indices = list(range(80))
            model, criterion, optimizer, scheduler = engine._initialize_training_components(
                train_indices, dataset
            )
            
            # Check optimizer type
            optimizer_type = type(optimizer).__name__
            print(f"  ✓ Successfully created {optimizer_type} optimizer")
            
            # Verify optimizer parameters
            param_groups = optimizer.param_groups[0]
            print(f"  ✓ Learning rate: {param_groups['lr']}")
            print(f"  ✓ Weight decay: {param_groups['weight_decay']}")
            
            if 'momentum' in param_groups:
                print(f"  ✓ Momentum: {param_groups['momentum']}")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return False
    
    print("\n✓ All optimizer options working correctly!")
    return True

if __name__ == "__main__":
    success = test_optimizer_options()
    sys.exit(0 if success else 1)
