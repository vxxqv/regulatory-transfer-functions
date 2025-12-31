# Compositionality eligibility audit

No screened dataset passed the frozen confirmatory gates. H1-H6 are unavailable, not negative biological results. No external expression matrix was downloaded or opened, and no response model, permutation, bootstrap or significance test was run.

The screen retains 58 scPerturb catalog records, all 54 files in Zenodo release 13350497, and four original-repository design records. Their union has 67 records. These are inventory records, not 67 independent studies; states, intervention subsets and overlapping releases are kept distinct. `results/dataset_eligibility.tsv` contains the full denominator, source links, available licenses, provider checksums, unresolved metadata and exclusions. Provider checksums were not verified against full expression releases.

## Assignment-level gates

Cell eligibility requires at least 50 cells for the pair and each matched single, plus 500 non-targeting controls. Confirmatory dataset eligibility additionally requires 40 pairs, 15 targets and 20 CD4 targets. The pair-model gate requires 100 pairs, 30 targets and 15 pairs per test fold. Matching genes, biological uncertainty units and 200 expressed overlapping genes remain required; passing cell counts alone is insufficient.

| Metadata subset | Double-target labels | Cell-eligible pairs | Eligible targets | Exact CD4 overlap | NTC cells |
|---|---:|---:|---:|---:|---:|
| Tian iPSC | 311 | 101 | 26 | 10 | 10,687 |
| Tian day-7 neurons | 311 | 82 | 26 | 10 | 15,580 |
| Replogle 2020 exp6 | 37 | 36 | 26 | 10 | 4,563 |
| Sunshine 2023 | 8,609 | 19 | 31 | 15 | 3,536 |
| Pacalin CRISPRi | 2 | 2 | 4 | 0 | 3,944 |
| Pacalin CRISPRa | 14 | 7 | 6 | 2 | 3,944 |
| Norman CRISPRa | 131 | 127 | 71 | 27 | 9,702 |
| Wessels Cas13 | 158 | 0 | 0 | 0 | 424 |
| Yao knockdown | 5,119 | 0 | 0 | 0 | 476 |
| Gasperini low-MOI, TSS only | 701 | 0 | 0 | 0 | 369 |

Zero eligible targets in the last three rows means that no pair passed all cell gates, not that the experiment lacked targeted genes. Higher-order assignments are excluded from singles and doubles. Pacalin subsets share controls and are not independent replications. Catalog counts and incidental multi-guide assignments are not treated as validated combinatorial designs.

Tian states have insufficient CD4 overlap and unresolved independent guide-assignment quality. Replogle has only 37 distinct gene pairs after guide-alias collapse, of which 36 meet cell gates. Sunshine has inadequate pair and CD4 overlap counts; its catalog and processing script also disagree on intervention type. Yao and the audited Gasperini subset fail the strict non-targeting-control gate. Gasperini high-MOI and at-scale gene-only pairing remain unverified, not proven absent. Adamson epistasis and the ARX/LMO1 cortical design have too few distinct target pairs. Same-gene dual guides are not distinct-gene double perturbations.

Norman passes broad cell-count gates but is CRISPRa, not a confirmatory knockdown dataset. Only 25 eligible pairs have both components represented in CD4, below the frozen pair-model requirement. The sensitivity branch was stopped before expression acquisition. Cas13, knockout, overexpression and chemical interventions cannot satisfy the confirmatory gate and were not pooled with CRISPRi.

## Mapping and exclusions

Unordered gene pairs use `target_a|target_b|accession`, with lexicographically sorted symbols. Each receives the frozen five-fold SHA256 assignment. The same pair across Tian states has the same fold. `observation_key` additionally includes dataset/state and is unique. No model was trained, so these are candidate folds, not a completed validation.

Replogle records require good coverage, one cell and two guides. `FDPS_2` and `HUS1_2` are collapsed only after confirming the same Ensembl IDs as their unsuffixed symbols. Norman records require good coverage and singlets; the repeated construct label and optional barcode suffix are validated before removing non-targeting components. Yao safe-targeting and unassigned cells are excluded from the NTC baseline. Pacalin activation, inhibition, mixed interventions and enhancer labels are separated. Gasperini retains only exact TSS-target and scrambled-control labels without enhancer or unassigned co-perturbations.

CD4 overlap uses exact symbols in the frozen 6,105-target primary universe after the documented Replogle alias correction. Unresolved legacy symbols are not guessed. For example, mapping both legacy Tian ATP synthase symbols could not raise its overlap from 10 to the required 20. Gene-expression overlap and remaining release-quality checks were not pursued after earlier gates failed.

## Reproduction and checks

From the repository root, with `REGULATORY_SOURCE_ROOT` pointing to the frozen source checkout:

```text
python -m analyses.compositionality.audit_metadata
python -m analyses.compositionality.screen_assignments
python -m analyses.compositionality.run_analysis
python -m pytest tests/test_compositionality.py -q
```

The default audit is offline. It verifies all 46 frozen input hashes and uses retained assignment metadata. `read_release_metadata.py` records the original bounded acquisition procedure for four releases. It accesses only observation metadata, not expression arrays. Range reads can include uninterpreted surrounding HDF5 bytes; full files were neither downloaded nor decoded. Exact guide IDs were present in the remote schemas but not extracted for these four subsets. Original metadata may contain unused historical annotations; a Norman design preview included a bulk-fitness column that was excluded from selection and is not an input to this audit.

`results/candidate_pair_metadata.tsv` reports every observed candidate pair and each failed cell gate. The remaining tables preserve model availability, H1-H6 decisions and every unperformed inference branch. Source checksums and the stopping decision are machine-readable. Focused tests independently reconstruct Norman and Replogle counts, confirm frozen thresholds and folds, and verify CD4 overlap and the absence of expression acquisition.

S27 was not created: a figure of this eligibility boundary would duplicate the tables without adding a scientific result. No PNG, PDF or SVG is therefore required or claimed. This audit does not establish additivity, saturation, synergy, regulatory emergence or failure of those mechanisms.
