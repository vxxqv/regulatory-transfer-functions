# KOLF2.1J access feasibility

## Decision

STOP. No currently published route satisfies the frozen integrity, outcome-blind selection, cell-count, control, guide-resolution, and resource gates without transferring the complete 189,393,177,972-byte object.

The complete Figshare object is scientifically lossless and could support H18 to H20. An initial audit snapshot found 182,763,712,512 free bytes. That command did not preserve an exact retrieval instant, so the value is retained only as an earlier snapshot taken before the audit commit at 2026-09-13T20:57:02Z. A timestamped confirmation retrieved at 2026-09-13T20:58:49.9958938Z found 182,568,767,488 free bytes, which was 6,824,410,484 bytes less than the file before analysis outputs or temporary storage were considered. The other existing routes are smaller or remotely addressable but omit required information or cannot establish identity with the frozen provider object.

Free space is dynamic. Later free-space changes require rechecking before any download, but they do not alter this audited STOP because both recorded snapshots were below the complete object size and neither left working space for outputs.

This is a resource and integrity stop, not a negative biological result. No expression matrix, response vector, differential-expression value, outcome ranking, or result figure was opened.

## Novelty and results gate

KOLF2.1J remains a valuable untouched system because it would test transfer outside the CD4, K562, RPE1, Th1, and Th2 systems. A valid result could materially strengthen or falsify H18 to H20. Convenience alone is insufficient: a lighter route must retain every frozen target eligible after the 50-cell rule, all response genes, non-targeting controls, guide identities, batch labels, and uncertainty needed for calibration and coverage.

The audit stopped once the finite official and documented route set was resolved. No alternate dataset or endpoint was substituted.

## Route findings

| Route | Decision | Finding |
|---|---|---|
| Figshare complete QC h5ad | STOP for audited resources | Version 1 file 64650261 is the sole complete non-strength-selected processed object. Figshare reports both supplied and computed MD5 `afd30fde1e6ad32969c29868394385d1`. A full read is required for the frozen local integrity gate. The timestamped free-space snapshot is recorded in the route table. |
| Figshare remote ranges | STOP | The public API documents file download by full HTTP GET. A header-only request returned 403, and no official range or array-subset interface is documented. Sparse HDF5 access cannot be validated without opening the outcome object. |
| Figshare strong subset | STOP | The 46,718,752,086-byte object contains 1,655 perturbations selected using observed perturbation strength. It is forbidden by the frozen selection rule. |
| Figshare pilot QC objects | STOP | The chromatin and metabolic files are distinct targeted pilot screens, not alternate encodings of the locked pan-genome population. |
| SRA reconstruction | STOP | PRJNA1173491 reports 16.72 TB of raw sequence. Some sequence access requires external permission, and reconstruction would create a new processed object rather than reproduce the frozen release exactly. |
| Author web row matrix | STOP | The author script documents 10,167 post-basic-QC targets by 38,606 genes stored as int8 NTC z scores with scale 0.04. It omits control cells, cell counts, guides, batches, and uncertainty. |
| Author top-gene summaries | STOP | Top-50 lists and histograms are truncated summaries and cannot support absolute prediction, calibration, or coverage. |
| Author internal Parquet | STOP | The build script references an internal aggregate Parquet that is not published. Its aggregate structure would still omit the cell and guide information required here. |
| Hugging Face Xet mirror | STOP | The mirror exposes byte ranges, the same byte count, LFS SHA256 `3e7b0eaae92cc4aacc1d85f6a416998d44445dd296aa44dab4f39d16de0dec79`, and revision `3d63c3832205c0ff6ea483b5732372de44c2ac43`. It is not author or institution attested, and no verified cross-hash links its object to the official Figshare MD5 without a complete read. |
| Perturb-Seqr signatures | STOP | The institutional download page advertises a 45,605,170-byte GMT and full limma-voom signatures. The full-signature URL currently returns 404. The available GMT lacks cells, controls, cell counts, uncertainty, and guides. |
| UVA Dataverse release metadata | STOP | Version 1.0 file 120749 is a 30,921-byte metadata archive. It references an external h5ad with MD5 `07e21fbfe0facaf5080a041c2350272c`, which differs from the frozen Figshare object and has no published SHA256 or subset service. |
| Provider-prepared frozen subset | STOP pending provider object | This is the only lower-transfer route that could retain the design. No such immutable provider-attested object is currently published. |

The machine-readable table contains the full 12-route denominator, integrity identifiers, transfer and storage expectations, losslessness assessment, selection risk, endpoint support, and exact blocker for each route.

## Integrity and opening boundary

Reading the h5ad superblock, group metadata, `.obs`, `.var`, or sparse-array chunk map requires GET requests against the sealed data object. Even if expression values are not intentionally decoded, this crosses the current unopened-object boundary before the role and integrity gate is complete. HDF5 structure inspection is therefore not permitted in this audit.

The official Figshare metadata are sufficient to freeze the article version, file identifier, byte count, filename, and provider MD5. They are not a local SHA256. The secondary mirror provides a content-addressed SHA256, but it does not provide a trusted cross-hash statement binding that object to the frozen Figshare file. Sampling ranges would not prove whole-object identity.

## Reopening condition

The branch may advance only if one of these conditions is met before outcome access:

1. Adequate storage and working space are made available for the complete Figshare object, followed by one-pass local SHA256 and MD5 verification.
2. The provider publishes an immutable lossless h5ad or Zarr subset containing the exact 5,084 frozen mapped targets, every cell for those targets, every non-targeting cell, all genes, guide and batch labels, and a manifest that binds the subset to Figshare file 64650261 and its provider MD5.

The provider subset must be generated from the frozen target list and the fixed below-50-cell exclusion rule, never from response magnitude, significance, model error, or perturbation strength. Its checksum, byte count, export code or exact export rule, role, mapping, and target splits must be locked before outcome inspection.

If the provider subset contains guide-level cell assignments and all non-targeting cells, the same download can support external guide-held-out information retention. It cannot support donor-held-out or two-state claims because KOLF2.1J is a single cell line in one matched state.

## Sources

1. Figshare version 1 record and complete file inventory: https://api.figshare.com/v2/articles/27261219
2. Figshare public download documentation: https://docs.figshare.com/old_docs/api/articles/
3. KOLF2.1J analysis repository and author web-data builders: https://github.com/y-doctor/KOLF2.1J_Perturbation_Cell_Atlas
4. NCBI BioProject PRJNA1173491: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1173491
5. KOLF2.1J publication: https://doi.org/10.1038/s41587-026-03199-w
6. UVA Dataverse release: https://doi.org/10.18130/V3/HIGT4C
7. Perturb-Seqr download record: https://perturbseqr.maayanlab.cloud/download
8. Secondary mirror metadata: https://huggingface.co/datasets/Boom5426/KOLF
