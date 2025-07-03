import numpy as np
import os
import glob
import pandas as pd
import random

import matplotlib.pyplot as plt

# Read the metadata CSV
meta_df = pd.read_csv('database/meta/final_specs.csv')

# Get unique class IDs
unique_classes = meta_df['class_id'].unique()

# Select 3 random classes
selected_classes = random.sample(list(unique_classes), min(3, len(unique_classes)))

# Select one random file from each selected class
selected_files = []
for class_id in selected_classes:
    class_files = meta_df[meta_df['class_id'] == class_id]['filename'].tolist()
    selected_file = random.choice(class_files)
    selected_files.append(selected_file)

# Create subplots for visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, filename in enumerate(selected_files):
    # Load the spectrogram data
    file_path = f'database/specs/{filename}'
    spec_data = np.load(file_path)
    
    # Get class ID for title
    class_id = meta_df[meta_df['filename'] == filename]['class_id'].iloc[0]
    
    # Plot the spectrogram
    im = axes[i].imshow(spec_data, aspect='auto', origin='lower', cmap='inferno')
    axes[i].set_title(f'Class {class_id}\n{filename}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Frequency')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()