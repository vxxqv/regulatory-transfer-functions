# Real-World GRN Benchmark

The benchmark uses the official BEELINE v4 experimental single-cell datasets and matched cell-type ChIP-seq networks. The fixed panel is hESC, hHep, mDC, mESC, and mHSC-E. Gold-standard edges are not used for gene selection, model fitting, or hyperparameter tuning.

Four fixed procedures are compared: absolute Pearson correlation, pseudotime-lagged correlation, ridge transition coefficients, and a stable finite-horizon transfer operator. The output reports every candidate TF-target edge, AUROC, AUPRC, early precision ratio, target-bootstrap intervals, runtime, and memory for all datasets and methods.
