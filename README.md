# Regulatory Transfer Functions

This repository develops a reproducible framework for measuring how regulatory perturbations propagate through human immune-cell gene networks. The primary analysis uses public causal perturbation data, with natural-genetic, chromatin, and disease evidence reserved for independent validation.

The study separates confirmatory analyses from exploratory extensions. Every figure is generated from frozen machine-readable data, and every input is recorded with its source, release, checksum, retrieval date, and permitted use.

## Status

Analysis in progress. Numerical results and manuscript claims remain provisional until all prespecified validation gates pass.

## Structure

- `config`: frozen analysis settings and statistical families
- `data_manifest`: source inventory, checksums, and access decisions
- `workflow`: reproducible workflow definitions
- `src`: ingestion, quality control, models, statistics, and visualization
- `analyses`: primary, sensitivity, negative-control, and replication analyses
- `figures`: main and supplementary figure programs and figure data
- `tables`: manuscript and supplementary tables
- `tests`: unit, schema, statistical, and figure checks
- `docs`: protocol, claim ledger, novelty audit, and reproduction notes

## Author

Vivaan Patni, Independent Researcher

## License

Code is released under the MIT License. Data remain governed by their source licenses and access terms.
