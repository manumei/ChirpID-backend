# Grid Searching

**Discarded Qualitative/Boolean Options:**

- Optimizer: Discard SGD and variants (performed consistently worse). Use only Adam (or optionally AdamW if available).

- Standardize: Discard "no standardization"—almost all top configs used standardization. Only keep standardize=True.

- Class Weights: Discard configs with use_class_weights=False. All top configs use class weights.

- Spec Augment / Noise Augment: Discard configs with both augmentations disabled. Keep at least spec_augment=True or noise_augment=True; best configs used spec_augment.

- LR Schedule: Discard cosine and "no schedule". Retain exponential and plateau. (Cosine performed inconsistently; best results with exponential.)

- Mixed Precision / AMP / Gradient Clipping: If hardware supports, keep mixed_precision=True and use gradient_clipping, as most top configs were AMP-optimized. If not, default to regular precision.

**Quantitative Parameter Ranges to Keep:**

- Batch Size: 20–44 (best configs cluster 24, 28, 32, 40, 44). Discard extreme values (<16 or >64).

- Initial LR: 0.001–0.004 (best results in this range). Discard ultra-high or ultra-low initial LR.

- L2 Regularization: 1e-4–3e-4 (avoid 1e-2, 1e-5, or 0).

- Exponential LR Decay Gamma: 0.95–0.98.

- Early Stopping Patience: 30–60.

- Num Epochs: 150–250 is sufficient; avoid ultra-long training runs.
