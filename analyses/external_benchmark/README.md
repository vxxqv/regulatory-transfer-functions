# External Benchmark and Falsification Challenge

This analysis compares frozen CD4-derived transfer features with prespecified baselines on an independent perturbation screen. `config/external_benchmark.yaml` defines eligibility, endpoints, folds, model families, ablations, nulls, uncertainty, and decision rules. `freeze_manifest.json` records the exact CD4 artifacts and hashes locked before holdout processing.

The previously inspected Replogle K562 comparison is retained as external replication. It is not described as untouched. Replogle RPE1 is reserved as the confirmatory falsification holdout. Target-level outputs include every eligible target, and exclusions retain one reason per excluded target.

BEELINE v4 provides a separate real-world GRN recovery stress test using experimental single-cell expression and matched ChIP-seq reference networks. Gold-standard edges are used only for final scoring.
