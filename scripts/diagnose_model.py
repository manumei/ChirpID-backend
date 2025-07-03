#!/usr/bin/env python3
"""
Model Diagnostic Script for ChirpID Backend

This script helps diagnose issues with the PyTorch model file that might cause
the "invalid load key, 'v'" error. It performs comprehensive checks and provides
solutions for common problems.

Usage:
    python diagnose_model.py [model_path]

If no model_path is provided, it will use the default location.
"""

import os
import sys
import hashlib
import pathlib
import tempfile
import argparse
from typing import Optional

def check_file_integrity(file_path: str) -> dict:
    """Check basic file integrity and properties."""
    result = {
        "exists": False,
        "readable": False,
        "size_bytes": 0,
        "size_mb": 0.0,
        "md5_hash": None,
        "first_16_bytes": None,
        "is_pytorch_format": False,
        "file_type": "unknown"
    }
    
    if not os.path.exists(file_path):
        return result
    
    result["exists"] = True
    result["readable"] = os.access(file_path, os.R_OK)
    
    if not result["readable"]:
        return result
    
    try:
        # Get file size
        size = os.path.getsize(file_path)
        result["size_bytes"] = size
        result["size_mb"] = size / (1024 * 1024)
        
        # Read first bytes and calculate hash
        with open(file_path, 'rb') as f:
            content = f.read()
            result["first_16_bytes"] = content[:16].hex()
            result["md5_hash"] = hashlib.md5(content).hexdigest()
            
            # Check if it's a PyTorch file (zip format)
            if content.startswith(b'PK'):
                result["is_pytorch_format"] = True
                result["file_type"] = "pytorch/zip"
            elif content.startswith(b'\x80\x03'):  # Pickle format
                result["file_type"] = "pickle"
            else:
                result["file_type"] = "unknown"
                
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return result

def test_pytorch_loading(file_path: str) -> dict:
    """Test PyTorch model loading with different methods."""
    result = {
        "pytorch_available": False,
        "pytorch_version": None,
        "load_default": False,
        "load_weights_only_true": False,
        "load_weights_only_false": False,
        "load_with_pickle": False,
        "error_messages": []
    }
    
    try:
        import torch
        result["pytorch_available"] = True
        result["pytorch_version"] = torch.__version__
        print(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        result["error_messages"].append(f"PyTorch not available: {e}")
        return result
    
    # Test different loading methods
    device = torch.device('cpu')  # Use CPU to avoid GPU issues
    
    # Method 1: Default loading
    try:
        checkpoint = torch.load(file_path, map_location=device)
        result["load_default"] = True
        print("✓ Default torch.load() succeeded")
    except Exception as e:
        result["error_messages"].append(f"Default load failed: {e}")
        print(f"✗ Default torch.load() failed: {e}")
    
    # Method 2: weights_only=True
    try:
        checkpoint = torch.load(file_path, map_location=device, weights_only=True)
        result["load_weights_only_true"] = True
        print("✓ torch.load() with weights_only=True succeeded")
    except Exception as e:
        result["error_messages"].append(f"Load with weights_only=True failed: {e}")
        print(f"✗ torch.load() with weights_only=True failed: {e}")
    
    # Method 3: weights_only=False explicitly
    try:
        checkpoint = torch.load(file_path, map_location=device, weights_only=False)
        result["load_weights_only_false"] = True
        print("✓ torch.load() with weights_only=False succeeded")
    except Exception as e:
        result["error_messages"].append(f"Load with weights_only=False failed: {e}")
        print(f"✗ torch.load() with weights_only=False failed: {e}")
    
    # Method 4: with pickle module
    try:
        import pickle
        checkpoint = torch.load(file_path, map_location=device, pickle_module=pickle)
        result["load_with_pickle"] = True
        print("✓ torch.load() with pickle_module succeeded")
    except Exception as e:
        result["error_messages"].append(f"Load with pickle_module failed: {e}")
        print(f"✗ torch.load() with pickle_module failed: {e}")
    
    return result

def suggest_solutions(file_check: dict, pytorch_check: dict) -> list:
    """Suggest solutions based on diagnostic results."""
    solutions = []
    
    if not file_check["exists"]:
        solutions.append("Model file does not exist. Ensure the file is copied to the correct location.")
        return solutions
    
    if not file_check["readable"]:
        solutions.append("Model file is not readable. Check file permissions: chmod 644 <model_file>")
    
    if file_check["size_bytes"] == 0:
        solutions.append("Model file is empty. Re-copy the model file from the source.")
        return solutions
    
    if not file_check["is_pytorch_format"] and file_check["file_type"] == "unknown":
        solutions.append(
            "File does not appear to be in PyTorch format. Verify this is the correct model file."
        )
    
    if not pytorch_check["pytorch_available"]:
        solutions.append("PyTorch is not available. Install PyTorch: pip install torch")
        return solutions
    
    if not any([
        pytorch_check["load_default"],
        pytorch_check["load_weights_only_true"],
        pytorch_check["load_weights_only_false"],
        pytorch_check["load_with_pickle"]
    ]):
        solutions.extend([
            "All PyTorch loading methods failed. This suggests:",
            "1. Model file is corrupted - re-copy the file from source",
            "2. PyTorch version incompatibility - check if model was saved with different PyTorch version",
            "3. Model was saved with different Python version",
            "4. File system corruption"
        ])
        
        # Check for specific error patterns
        error_text = " ".join(pytorch_check["error_messages"]).lower()
        if "load key 'v'" in error_text:
            solutions.extend([
                "",
                "The 'load key v' error specifically indicates:",
                "• PyTorch version mismatch (most common)",
                "• Corrupted pickle/zip structure in the file",
                "• Try re-saving the model with same PyTorch version as server"
            ])
    
    return solutions

def main():
    parser = argparse.ArgumentParser(description="Diagnose PyTorch model loading issues")
    parser.add_argument("model_path", nargs="?", help="Path to the model file to diagnose")
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Use default path relative to script location
        script_dir = pathlib.Path(__file__).resolve().parent
        repo_root = script_dir.parent
        model_path = repo_root / "models" / "bird_cnn.pth"
        model_path = str(model_path)
    
    print("=" * 60)
    print("ChirpID Model Diagnostic Tool")
    print("=" * 60)
    print(f"Diagnosing model file: {model_path}")
    print()
    
    # File integrity check
    print("1. File Integrity Check")
    print("-" * 30)
    file_check = check_file_integrity(model_path)
    
    for key, value in file_check.items():
        if key == "first_16_bytes" and value:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # PyTorch loading test
    if file_check["exists"] and file_check["readable"]:
        print("2. PyTorch Loading Test")
        print("-" * 30)
        pytorch_check = test_pytorch_loading(model_path)
        print()
        
        # Summary
        print("3. Summary")
        print("-" * 30)
        working_methods = sum([
            pytorch_check["load_default"],
            pytorch_check["load_weights_only_true"],
            pytorch_check["load_weights_only_false"],
            pytorch_check["load_with_pickle"]
        ])
        
        if working_methods > 0:
            print(f"✓ {working_methods} loading method(s) work successfully")
        else:
            print("✗ No loading methods work")
        
        print()
        
        # Solutions
        print("4. Suggested Solutions")
        print("-" * 30)
        solutions = suggest_solutions(file_check, pytorch_check)
        for i, solution in enumerate(solutions, 1):
            if solution.strip():
                print(f"{i}. {solution}")
            else:
                print()
    else:
        print("Skipping PyTorch tests due to file access issues.")
        print()
        print("Suggested Solutions:")
        print("-" * 30)
        solutions = suggest_solutions(file_check, {})
        for i, solution in enumerate(solutions, 1):
            print(f"{i}. {solution}")

if __name__ == "__main__":
    main()
