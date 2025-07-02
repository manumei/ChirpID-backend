#!/usr/bin/env python3
"""
Test script to verify the fixes for BirdCNN_v4, BirdCNN_v8, and BirdCNN_v13
"""

import torch
import torch.nn as nn
from utils.models import BirdCNN_v4, BirdCNN_v8, BirdCNN_v13

def test_model_forward_pass(model_class, model_name, num_classes=10):
    """Test forward pass of a model architecture"""
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    try:
        # Create model instance
        model = model_class(num_classes=num_classes)
        model.eval()
        
        # Create dummy input (batch_size=4, channels=1, height=224, width=313)
        dummy_input = torch.randn(4, 1, 224, 313)
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ {model_name} forward pass successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output shape: torch.Size([4, {num_classes}])")
        
        # Verify output shape
        if output.shape == torch.Size([4, num_classes]):
            print(f"   ✅ Output shape is correct!")
        else:
            print(f"   ❌ Output shape mismatch!")
            return False
            
        # Test backward pass (gradient computation)
        model.train()
        output = model(dummy_input)
        loss = output.sum()  # Simple loss for testing
        loss.backward()
        print(f"   ✅ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return False

def main():
    """Main testing function"""
    print("TESTING FIXED MODEL ARCHITECTURES")
    print("=" * 80)
    
    # Number of classes for testing
    num_classes = 10
    
    # Models to test
    models_to_test = [
        (BirdCNN_v4, "BirdCNN_v4"),
        (BirdCNN_v8, "BirdCNN_v8"), 
        (BirdCNN_v13, "BirdCNN_v13")
    ]
    
    results = {}
    
    # Test each model
    for model_class, model_name in models_to_test:
        results[model_name] = test_model_forward_pass(model_class, model_name, num_classes)
    
    # Summary
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    
    successful = 0
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_name:<15}: {status}")
        if success:
            successful += 1
    
    print(f"\nOverall: {successful}/{len(models_to_test)} models passed")
    
    if successful == len(models_to_test):
        print("🎉 All fixes successful! Models are ready for training.")
    else:
        print("⚠️  Some models still have issues. Check the error messages above.")

if __name__ == "__main__":
    main()
