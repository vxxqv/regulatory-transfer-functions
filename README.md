# Regulatory Transfer Functions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22804223.svg)](https://doi.org/10.5281/zenodo.22804223)

This repository contains the code and frozen data products used to estimate transcriptional propagation after CRISPR interference in primary human CD4 T cells. The workflow separates proximal perturbation strength from distal response, quantifies signed programme rerouting across cell states, and tests portability with external perturbation, regulatory network, molecular, and genetic evidence.

## Contents

- `config` contains frozen model, validation, and sensitivity settings.
- `data_manifest` records source accessions, versions, checksums, and redistribution decisions.
- `workflow` and `src` contain ingestion, quality control, modelling, and statistical utilities.
- `analyses` contains executable analysis blocks and frozen machine readable results.
- `tables` contains compact result tables with complete denominators and exclusions.
- `tests` checks schemas, estimands, resampling, causal gates, and release integrity.
- `docs` contains the protocol, data dictionary, evidence graph, glossary, and reproduction guide.

## Reproduction

Create the environment from `environments/environment.yml` or `environments/requirements.txt`, then run the focused checks from the repository root.

```text
python -m pytest
```

Large upstream datasets are not duplicated. Their acquisition records and integrity checks are listed in `data_manifest/sources.tsv`. Frozen processed results are included where source terms permit redistribution.

Version 1.0.0 is archived at [Zenodo](https://doi.org/10.5281/zenodo.22804223).

## Author

Vivaan Patni, Independent Researcher

## License

Code is available under the MIT License. Source datasets remain subject to their original licenses and access terms.
