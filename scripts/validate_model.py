#!/usr/bin/env python3
"""
Quick model validation script for Ubuntu server.
Run this script on your Ubuntu server to test if the model can be loaded.

Usage:
    python validate_model.py
"""

import os
import sys
import pathlib

def main():
    print("ChirpID Model Validation")
    print("=" * 40)
    
    # Get model path
    script_dir = pathlib.Path(__file__).resolve().parent
    if script_dir.name == "scripts":
        repo_root = script_dir.parent
    else:
        repo_root = script_dir
    
    model_path = repo_root / "models" / "bird_cnn.pth"
    
    print(f"Repository root: {repo_root}")
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("❌ Model file not found!")
        return False
    
    print(f"Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Test PyTorch availability
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False
    
    # Test model loading
    print("\nTesting model loading...")
    device = torch.device('cpu')
    
    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        print(f"✅ Model loaded successfully!")
        print(f"   Type: {type(checkpoint)}")
        
        if hasattr(checkpoint, 'keys'):
            print(f"   Keys: {list(checkpoint.keys())[:5]}...")  # First 5 keys
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        
        # Try alternative methods
        print("\nTrying alternative loading methods...")
        
        methods = [
            ("weights_only=True", lambda: torch.load(str(model_path), map_location=device, weights_only=True)),
            ("with pickle", lambda: torch.load(str(model_path), map_location=device, pickle_module=__import__('pickle'))),
        ]
        
        for method_name, load_func in methods:
            try:
                checkpoint = load_func()
                print(f"✅ Success with {method_name}")
                return True
            except Exception as method_error:
                print(f"❌ Failed with {method_name}: {method_error}")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
