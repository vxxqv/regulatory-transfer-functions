# External Benchmark and Falsification Challenge

This analysis compares frozen CD4-derived transfer features with prespecified baselines on an independent perturbation screen. `config/external_benchmark.yaml` defines eligibility, endpoints, folds, model families, ablations, nulls, uncertainty, and decision rules. `freeze_manifest.json` records the exact CD4 artifacts and hashes locked before holdout processing.

The previously inspected Replogle K562 comparison is retained as external replication. It is not described as untouched. Replogle RPE1 is reserved as the confirmatory falsification holdout. Target-level outputs include every eligible target, and exclusions retain one reason per excluded target.

BEELINE v4 provides a separate real-world GRN recovery stress test using experimental single-cell expression and matched ChIP-seq reference networks. Gold-standard edges are used only for final scoring.

The RPE1 holdout contains 490 targets shared with the frozen CD4 target set. The full transfer model achieved R2 = 0.071 versus 0.030 for simple covariates, but the paired target-bootstrap delta was unresolved. Removing the module-vector block improved R2 to 0.191. Matched CD4-feature permutations were not exceeded. The observed CD4 module-neighbor topology exceeded 1,000 directed degree-preserving rewirings after correction. All model, ablation, calibration, coverage, cost, target, exclusion, bootstrap, permutation, and rewiring results are retained in `results/`.
