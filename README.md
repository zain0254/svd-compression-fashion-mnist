# Rank-k SVD Compression of Images and Neural-Network Weights

Semester project, Mathematical and Statistical Foundations for Data Science & Computational Intelligence (Fall 2026).
Muhammad Zain Hassan, 26i-8015, MSDS Section 1A.

**Question:** how does the retained rank k change reconstruction error and classification accuracy when SVD is applied
(A) to the images and (B) to the first-layer weights of a small trained network?

## Files
- `svd_compression_checkpoint_26i-8015.ipynb`: full pipeline (Parts A and B), executed on UCI digits.
- `results_digits.png`: singular value spectrum and accuracy versus k.
- `requirements.txt`: Python packages.

## How to run
1. `pip install -r requirements.txt`
2. Open the notebook and set `DATASET = "fashion"` (Fashion-MNIST from OpenML, 50k/10k/10k split) or `"digits"` (UCI digits, 60/20/20 split).
3. Run all cells. CPU only; digits takes under a minute.

## Reproducibility
Seed 42. The mean and SVD are fitted on training data only. Rank k is the only setting that changes.
Only validation results are reported; the test set is used once, for the final paper.
