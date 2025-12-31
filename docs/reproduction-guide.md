# Reproduction guide

## Scope

The repository contains frozen processed outputs for reproducing the manuscript tables and figures on a standard laptop. Large public source files are not duplicated. Acquisition scripts and the data manifest identify those inputs when a full rebuild is required.

## Environment

Create the declared Python environment from `environments/environment.yml` or install the pinned packages in `environments/requirements.txt`. Run commands from the repository root. Do not change frozen configuration files for a confirmatory reproduction.

## Fast reproduction

1. Verify the source and freeze manifests.
2. Run the focused tests in `tests`.
3. Run each main plotting script in `figures/fig01` through `figures/fig07`.
4. Run the supplementary plotting scripts in `figures/supplement`.
5. Run `analyses/figure_qc/run_figure_qc.py`.
6. Run `manuscript/build_supplement.py`.

This path reads existing frozen results and does not redownload data or refit models.

## Full rebuild

The workflow file records the dependency order. A full rebuild requires the public source datasets listed in `data_manifest/sources.tsv`, substantially more time and memory, and access consistent with each source's terms. Confirmatory freeze manifests must be checked before processing the untouched RPE1 outcomes.

## Expected outputs

- Seven main figures in PNG, PDF, and SVG.
- Supplementary Figures S1-S14, S16-S21, and S23 in the same formats.
- Panel-level source tables for every figure.
- Supplementary Tables S1-S8.
- Automated figure QC and software-test reports.

Any numerical difference from the frozen outputs should be treated as a reproduction failure until explained by a documented platform or dependency difference.
