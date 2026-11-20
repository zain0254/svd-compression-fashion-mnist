# Rank-k SVD Compression of Images and Neural-Network Weights

Semester project, Mathematical and Statistical Foundations for Data Science & Computational Intelligence (Fall 2026).
Muhammad Zain Hassan, 26i-8015, MSDS Section 1A.

**Question:** how does the retained rank k change reconstruction error and classification accuracy when SVD is applied
(A) to the images and (B) to the first-layer weights of a small trained network?

## Files
| File | Contents |
|---|---|
| `svd_compression_full_experiments_26i-8015.ipynb` | Full experiments: Parts A and B, Eckart–Young check, plain vs data-aware SVD, frozen retraining, weight error vs error on real inputs, 5 seeds |
| `svd_compression_checkpoint_26i-8015.ipynb` | Feasibility checkpoint version (Parts A and B only) |
| `results_digits.png` | Spectrum, accuracy vs k, weight error vs error on real inputs (UCI digits) |
| `reconstructions_digits.png` | Validation images rebuilt at several k |
| `eigen_images_digits.png` | Top 8 singular vectors shown as images |
| `requirements.txt` | Python packages |

## How to run
1. `pip install -r requirements.txt`
2. Open the full-experiments notebook and set `DATASET = "fashion"` (Fashion-MNIST from OpenML, 50k/10k/10k split) or `"digits"` (UCI digits, 60/20/20 split).
3. Run all cells. CPU only. Digits takes seconds; Fashion-MNIST takes longer (the 5-seed network training is the slow part).

## Reproducibility
Seeds 42–46 for the network, 42 for splits and classifiers. The mean and every SVD are fitted on training data only.
Rank k is the only setting that changes. Only validation results are reported; the test set is used once, for the final paper.
