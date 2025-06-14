import os
import shutil

def clean_dir(dest_dir):
    ''' Deletes the raw audio files in the dest_dir.'''
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)