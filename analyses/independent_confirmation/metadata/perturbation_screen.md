# Independent perturbation metadata screen

## Scope and denominator

This is a metadata-only screen under the frozen 2026-09-13 protocol. No expression matrix, signed response vector, differential-expression table, statistical result, result figure, or ranking was opened. The inspected material was limited to bibliographic records, methods, repository records, file inventories, checksums, licenses, target and guide labels, cell-count summaries, controls, states, donors, and repeated-unit design.

The denominator contains 75 entries: all 67 entries in the existing scPerturb v1.4 outcome-blind audit plus 8 prespecified additions not already represented there. The candidate table has 19 rows: 18 individually detailed records and one pointer row covering the other 57 scPerturb entries. Individual identifiers and exclusion reasons for those 57 remain in `analyses/compositionality/results/dataset_eligibility.tsv`; their denominator weight is preserved in `perturbation_candidates.tsv`. Screening stopped after the named KOLF, Song, ARC, Nadig, Schmidt, Shifrut, GSE278572, GSE241882, FOXP3, and GSE314342 records were resolved.

Selection order was applied exactly as frozen: gate pass, independent cohort and provider provenance, mapped CD4 target count, common response-gene count, repeated-unit count, then accession. No expected performance or reported biological result entered selection.

## Gate decisions

| Gate | Decision | Basis |
|---|---|---|
| Portability metadata selection for H18-H20 | ADVANCE | KOLF2.1J PRJNA1173491 is the sole stage-1 selection. It is human CRISPRi transcriptomics with a full processed count release, non-targeting controls, unambiguous guide and target labels, 5,084 exact frozen CD4 target overlaps, at least 500 transcriptome genes, and a published coverage distribution that guarantees a large eligible subset. |
| Portability outcome opening | STOP | The 189,393,177,972-byte full h5ad must receive a local full-file SHA256, the provider MD5 must be verified, per-target exclusions below 50 cells must be frozen, and the role and mapping must be locked. |
| Definitive primary T-cell replication for H21-H22 | STOP | Zero datasets pass every per-dataset gate, so the required pair of independent eligible datasets is unavailable. |
| Guide-by-donor reconstruction for H23-H24 | STOP | Zero datasets establish three supported guide efficacies, at least three donors, at least 30 cells per guide-donor, at least 500 genes, controls, target-state mappings, and guide assignments together. |

H18-H20, H21-H22, and H23-H24 remain not tested. The decisions above authorize only stage-1 metadata selection, not downstream analysis.

## Selected portability record

The KOLF2.1J atlas is linked to [PRJNA1173491](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1173491), the [version-1 Figshare release](https://api.figshare.com/v2/articles/27261219), and the [author repository](https://github.com/y-doctor/KOLF2.1J_Perturbation_Cell_Atlas). The version of record is [10.1038/s41587-026-03199-w](https://doi.org/10.1038/s41587-026-03199-w). Figshare reports CC-BY-4.0 for the data and MD5 `afd30fde1e6ad32969c29868394385d1` for `KOLF_Pan_Genome_QC_Filtered.h5ad`; the author code repository is MIT licensed.

The author target list contains 11,693 symbols and identifiers. Exact uppercase-symbol matching to the frozen CD4 target list gives 5,084 overlaps with mapping SHA256 `c22383a1e28fed9cd2e7c667c711a997f9373fd80c8f909628f150d21a8c481a`. The paper reports three guides per target, 478 original non-targeting guides, and more than 100 cells for 88 percent of perturbations. Even allowing the reported percentage to be rounded, the dataset-level high-coverage subset can exceed 100 mapped targets after per-target filtering. This is dataset-level sufficiency only: every target below 50 cells must be excluded before any outcome is opened.

`KOLF_Strong_Perturbations.h5ad` was not used because its membership is defined from observed perturbation strength. Only the full non-outcome-selected file is eligible for the stage-2 checksum gate.

Two prespecified reuse flags are metadata-only. External guide-held-out information retention is ADVANCE because the atlas has three guides per target and target labels. Two-state conserved-core decomposition is STOP because the atlas has one matched cellular state.

## Principal stops

- Song Jurkat GSE249595/PRJNA1049794 has 5,995 exact frozen CD4 overlaps, four guides per target, 250 non-targeting guides, and two matched stimulation states. It is STOP because the high-MOI design has a median of 13 guides per cell, the reported 400 cells per gene is an average rather than a minimum, and no metadata-only author-validated pseudobulk uncertainty object establishes isolated target profiles.
- ARC H1 `arch1` is STOP under the novelty and result gate because the official design selected targets and train/validation/test splits using observed response strength and cell-quality results. It is not substituted despite adequate nominal scale.
- Nadig GSE264667 HepG2 and Jurkat are STOP. Processed single-cell populations and 5 percent non-targeting constructs are available, but the official metadata inventory does not establish the exact mapped target subset or the minimum of 50 cells per target without opening sealed cell-level data. Fixed paired same-gene guides and cell lines also fail guide-by-donor eligibility.
- Schmidt PRJNA787633/GSE190604 is CRISPRa sensitivity evidence only. It has resting and restimulated primary human T cells and about 70 targets, but only two donors; exact frozen mapping and the per-target-state minimum are not locked.
- Shifrut GSE119450 is STOP with fewer than 50 targets and two donors.
- GSE278572 is STOP with two donors, below the minimum of three repeated units.
- GSE241882 is STOP with one activation state and fewer than 50 targets.
- FOXP3 PRJDB16517 is STOP because the Perturb-icCITE-seq component has raw DDBJ sequencing and processing scripts but no processed transcriptome release, no two matched activation states, and no three repeated units in the public metadata.
- GSE314342 is the current development source and remains a permanent pre-freeze exclusion. It is not counted as independent evidence.

## Hash and access controls

Every candidate row has a SHA256 of the exact `provisional_metadata_hash_basis` field. These hashes cover the selection-driving accession, version, system, modality, availability, mapping, controls, guide, donor, state, cell, gene, decision, and novelty fields. The tests independently recompute every hash, require all three decisions, require current data-bearing artifacts to remain unopened, enforce the single provisional portability selection, and confirm the 75-entry denominator.

No outcome opening is authorized by this screen. No full 189 GB download should occur until a resource-cost feasibility check identifies a lossless, outcome-blind access route or the user explicitly approves the cost. That feasibility check does not replace any frozen control: the KOLF stage-2 gate remains STOP until the full-file SHA256, provider checksum verification, per-target mapping filter, immutable role, and selection lock are recorded.
