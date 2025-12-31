# Reproduction guide

## Scope

The repository contains frozen processed outputs for reproducing the reported analyses on a standard laptop. Large public source files are not duplicated. Acquisition scripts and the data manifest identify those inputs when a full rebuild is required.

## Environment

Create the declared Python environment from `environments/environment.yml` or install the pinned packages in `environments/requirements.txt`. Run commands from the repository root. Do not change frozen configuration files for a confirmatory reproduction.

## Fast reproduction

1. Verify the source and freeze manifests.
2. Run the focused tests in `tests`.
3. Run the analysis blocks listed in `docs/methods-code-crosswalk.md` as needed.
4. Compare regenerated outputs with the frozen result tables.

This path reads existing frozen results and does not redownload data or refit models.

## Full rebuild

The Snakefile records the core ingestion and primary-analysis dependency order. Extended analyses are run through the scripts listed in `docs/methods-code-crosswalk.md`; they are not all represented as Snakefile rules. A full rebuild requires the public source datasets listed in `data_manifest/sources.tsv`, substantially more time and memory, and access consistent with each source's terms. The historical external-benchmark freeze must be checked before reproducing the original RPE1 falsification analysis. RPE1 is now previously inspected and cannot serve as an untouched confirmation for later extensions.

## Expected outputs

- Frozen result tables for each analysis block.
- Complete denominators, exclusions, null results, and unavailable analyses.
- Supplementary Tables S1-S8.
- Software test reports.

Any numerical difference from the frozen outputs should be treated as a reproduction failure until explained by a documented platform or dependency difference.
