# ChirpID Backend - Model Loading Troubleshooting

## Problem: "invalid load key, 'v'" Error

This error occurs when PyTorch cannot load the model file (`bird_cnn.pth`). It typically happens due to:

1. **PyTorch version mismatch** (most common)
2. **Corrupted model file** during transfer
3. **File permissions** issues
4. **Incompatible Python/PyTorch versions**

## Quick Diagnosis

Run the diagnostic script to identify the issue:

```bash
cd /path/to/chirpid-backend
python scripts/diagnose_model.py
```

Or specify a custom model path:
```bash
python scripts/diagnose_model.py /path/to/your/bird_cnn.pth
```

## Manual Diagnosis Steps

### 1. Check File Integrity

```bash
# Check if model file exists
ls -la models/bird_cnn.pth

# Check file size (should be several MB, not 0 bytes)
du -h models/bird_cnn.pth

# Check file permissions (should be readable)
ls -la models/bird_cnn.pth

# Check first few bytes (should start with 'PK' for ZIP format)
head -c 16 models/bird_cnn.pth | xxd
```

### 2. Test PyTorch Loading

```python
import torch
import os

model_path = "models/bird_cnn.pth"

# Test basic loading
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print("✓ Model loads successfully")
    print(f"Type: {type(checkpoint)}")
except Exception as e:
    print(f"✗ Loading failed: {e}")
```

### 3. Check PyTorch Version

```bash
# On your local machine (where model works)
python -c "import torch; print('PyTorch version:', torch.__version__)"

# On Ubuntu server (where model fails)
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Common Solutions

### Solution 1: PyTorch Version Mismatch

If PyTorch versions differ between local machine and server:

**Option A: Upgrade server PyTorch to match local**
```bash
# Check local version first, then on server:
pip install torch==<your_local_version>
```

**Option B: Re-save model with server's PyTorch version**
```python
# On your local machine, save with compatible format
import torch
model = torch.load('bird_cnn.pth', map_location='cpu')
torch.save(model, 'bird_cnn_compatible.pth', _use_new_zipfile_serialization=False)
```

### Solution 2: File Corruption

If file was corrupted during transfer:

```bash
# Re-copy the model file using proper methods
# Method 1: SCP with verification
scp -C bird_cnn.pth user@server:/path/to/chirpid-backend/models/
md5sum bird_cnn.pth  # On local
md5sum /path/to/chirpid-backend/models/bird_cnn.pth  # On server
# MD5 hashes should match

# Method 2: RSYNC with checksum verification
rsync -avz --checksum bird_cnn.pth user@server:/path/to/chirpid-backend/models/
```

### Solution 3: File Permissions

```bash
# Fix permissions on Ubuntu server
chmod 644 models/bird_cnn.pth
chown $USER:$USER models/bird_cnn.pth
```

### Solution 4: Force Compatible Loading

Add this fallback in your code:

```python
def load_model_with_fallbacks(model_path, device):
    """Try multiple loading methods for compatibility."""
    
    # Method 1: Default (newest PyTorch)
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e1:
        print(f"Default load failed: {e1}")
        
        # Method 2: Legacy mode
        try:
            return torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e2:
            print(f"weights_only=True failed: {e2}")
            
            # Method 3: With explicit pickle
            try:
                import pickle
                return torch.load(model_path, map_location=device, pickle_module=pickle)
            except Exception as e3:
                print(f"All loading methods failed: {e1}, {e2}, {e3}")
                raise e1  # Raise original error
```

## Testing the Fix

1. **Run health check endpoint:**
```bash
curl http://localhost:5001/api/audio/health
```

2. **Test with sample audio:**
```bash
curl -X POST -F "file=@test_audio.wav" http://localhost:5001/api/audio/upload
```

3. **Check server logs:**
```bash
tail -f /path/to/your/server/logs
```

## Prevention

To prevent this issue in the future:

1. **Document PyTorch versions** used for training and deployment
2. **Use virtual environments** with pinned versions
3. **Include version checks** in deployment scripts
4. **Test model loading** as part of deployment verification
5. **Keep backup copies** of working model files

## Environment Setup

Create a `requirements.txt` with exact versions:

```
torch==1.13.1  # or your specific version
torchvision==0.14.1
numpy==1.21.0
pandas==1.5.0
# ... other dependencies
```

Install with:
```bash
pip install -r requirements.txt
```

## Contact

If none of these solutions work, please provide:

1. Output of `scripts/diagnose_model.py`
2. PyTorch versions on both local and server
3. Full error traceback
4. File sizes and checksums of model files
