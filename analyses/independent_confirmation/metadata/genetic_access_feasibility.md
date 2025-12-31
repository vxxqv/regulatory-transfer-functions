# Genetic access feasibility

## Decision

The five corrected immune-disease GWAS candidates remain unopened. All five are direct phenotypes with complete-file metadata and listed provider MD5 values. The external bundle is scientifically capable of competitive enrichment and complete locus screening, but it is not ready for a role lock. The current decision is `STOP_PENDING_PROTOCOL_AND_RESOURCE_LOCK` for all five candidates.

Storage is a resource-risk gate, not a demonstrated execution impossibility. The C drive has 182,568,120,320 free bytes, or 170.03 GiB. A sequential chromosome-wise plan using indexed compressed LD and eQTL files, one harmonized GWAS at a time, and the small H3K27ac matrix could plausibly peak near 100 to 140 GiB. This estimate excludes the 11.3 GB raw H3K27ac archive and requires immediate retirement of chromosome intermediates. The 220 to 300 GiB estimate is the conservative preferred workspace for parallel intermediates and retained audit copies. Exact baseline-LD and negative-control selections could change the peak, so opening still requires a frozen file-retention and peak-space plan.

The following items are resolvable before outcome opening:

1. Select an exact negative-disease GWAS by an outcome-blind protocol amendment.
2. Freeze matched negative-locus variables and tolerances.
3. Freeze the exact baseline-LD release, build, weights, and annotations.
4. Freeze hg19-to-GRCh38 liftover acceptance rules for GSE118189.
5. Compute and lock local full-file SHA256 values for eQTL Catalogue and GEO files whose providers do not publish content checksums.
6. Define QTS000040 subtype roles outcome-blind if that study is retained as a sensitivity resource.

Minor-allele-frequency and imputation-quality column availability is a post-opening schema gate, not a metadata-selection failure. Each candidate must stop immediately after checksum verification if those fields cannot support the frozen locus rule. The absence of exact 8h and 48h eQTL contexts is also not an opening failure. It limits state-resolved interpretation and must remain explicit.

No outcome-opening order is assigned because the protocol and storage gates are incomplete. Proxy diseases, nominal pathway tests, and post-outcome matching remain prohibited.

## Candidate denominator

| Disease | Accession | File size | Provider checksum | Decision |
| --- | --- | ---: | --- | --- |
| Asthma | GCST90475259 | 581,977,346 bytes | MD5 listed, unverified | Stop pending lock |
| Hashimoto's thyroiditis | GCST90627754 | 1,330,469,250 bytes | MD5 listed, unverified | Stop pending lock |
| Multiple sclerosis | GCST90475833 | 599,875,485 bytes | MD5 listed, unverified | Stop pending lock |
| Type 1 diabetes mellitus | GCST90475661 | 587,891,021 bytes | MD5 listed, unverified | Stop pending lock |
| Ulcerative colitis | GCST90476067 | 595,948,811 bytes | MD5 listed, unverified | Stop pending lock |

The five GWAS files total 3,696,161,913 bytes. Each metadata YAML reports GRCh38, GWAS-SSF v1.0, European ancestry, `is_harmonised: false`, and `is_sorted: false`. Harmonization and sorting would therefore be required after checksum verification. No GWAS data row or column header was opened.

## Shared resources

### LD reference

The exact candidate LD bundle is the 1000 Genomes NYGC 30x phased GRCh38 autosomal release dated 2022-04-22. It contains 22 VCF files and 22 tabix indexes. HTTP metadata give an exact combined size of 29,767,081,443 bytes. The official `20220804_manifest.txt` lists an MD5 value for every file. The intended ancestry subset is the 503 unrelated European samples from CEU, FIN, GBR, IBS, and TSI in `integrated_call_samples_v3.20130502.ALL.panel`, with pedigree filtering from `1kGP.3202_samples.pedigree_info.txt`.

The VCF checksums are listed but were not verified. The population panel and pedigree file expose only HTTP ETags, not provider content digests. Their local SHA256 values can be locked after download. No genotype file was opened.

### T-cell eQTL and fine mapping

The only exact timed file set identified without outcome inspection is Cytoimmgen QTS000041:

| State metadata | Dataset | Required files | Exact size |
| --- | --- | --- | ---: |
| Unstimulated 16h | QTD000689 | complete cis-eQTL, tabix index, credible sets, log Bayes factors | 3,055,105,437 bytes |
| Stimulated 16h | QTD000693 | complete cis-eQTL, tabix index, credible sets, log Bayes factors | 3,018,755,600 bytes |
| Stimulated 40h | QTD000690 | complete cis-eQTL, tabix index, credible sets, log Bayes factors | 3,659,882,958 bytes |

The exact QTS000041 set totals 9,733,743,995 bytes. All files are GRCh38 and distributed under CC BY 4.0. The release directory lists byte sizes and HTTP ETags but no provider content checksum. A local full-file SHA256 is therefore required before opening. Dataset-level ancestry is not declared in the release 7 metadata row and must be verified from study metadata before colocalization. Exact 8h and 48h states are absent, so these resources support state-relevant rather than time-exact inference.

Nathan QTS000040 is retained in the denominator but stopped before file selection. Its 29 release 7 gene-expression datasets cover CD4-related clusters from 249 donors, yet every condition label is `naive`. Selecting clusters such as activated, Th1, Th2, or Treg after viewing downstream concordance would violate the frozen role rule.

### Chromatin

GSE118189 provides a 116,752,142-byte ATAC count matrix across 175 primary immune-cell samples. The series includes resting and stimulated CD4 subsets, but the processed coordinates are hg19 and the release does not publish a checksum for the matrix. This is resolvable with a locked local SHA256 and prespecified liftover QC.

GSE244035 provides a 2,652,932-byte GRCh38 H3K27ac peak matrix and an 11,307,591,680-byte archive of per-sample BED and BEDGRAPH files across Rest and Stim48hr primary CD4 Teff and Treg samples. It is independent of the GSE314342 primary perturbation dataset. It has no 8h state and GEO publishes file sizes but no provider content checksum. The missing time point limits interpretation but does not fail the state-relevant chromatin opening gate.

The UCSC `hg19ToHg38.over.chain.gz` file is 227,698 bytes with provider MD5 `35887f73fe5e2231656504d1f6430900`. The chain alone does not define acceptable unmapped-interval loss, split mappings, or sensitivity rules.

### Falsification and baseline resources

The frozen five-candidate set contains no negative-disease GWAS. This audit does not expand the disease denominator, but a separate outcome-blind protocol amendment can select one before any file is opened. Matched negative loci also need fixed distance, gene density, LD, accessibility, MAF, and locus-size tolerances. The configuration names a baseline-LD covariate but does not identify a release, build, weight set, or annotation bundle. These are pending protocol locks rather than permanent availability failures.

## Resource estimate

The exact files listed in the feasibility table, excluding unresolved QTS000040 subtypes, total 54,624,319,078 bytes, or 50.873 GiB. The minimum bundle without the GSE244035 per-sample archive is 40.342 GiB.

A sequential disk-bounded plan would retain the 40.342 GiB minimum compressed bundle, omit the 11.3 GB per-sample H3K27ac archive, process one chromosome and one GWAS at a time, and retire disposable intermediates after their locked hashes and summaries are recorded. Allowing approximately 60 to 100 GiB for one active harmonized GWAS, chromosome-wise LD products, tool scratch space, and final outputs gives a provisional peak of 100 to 140 GiB. This is a planning estimate because exact baseline-LD and negative-control files are not yet selected. It must be replaced by a component-level peak-space budget before opening.

The conservative preferred workspace remains 220 to 300 GiB for downloaded files, decompressed or indexed intermediates, parallel chromosome-wise LD products, harmonized GWAS files, credible-set products, and retained audit copies. Transfer time is approximately 3 to 10 hours on a stable 25 to 100 Mbit/s connection after allowing for public-endpoint throttling. Chromosome-wise processing would require at least 32 GiB RAM, with 64 GiB recommended and 128 GiB safer for dense loci and sensitivity runs. Competitive enrichment, fine mapping, three-context colocalization, calibration, and frozen resampling across five diseases would require roughly 500 to 1,500 CPU-hours.

Current free storage is 170.03 GiB. It is above the provisional 100 to 140 GiB sequential peak but below the 220 to 300 GiB conservative preferred workspace. Execution is therefore not impossible on capacity grounds. It remains stopped until the exact baseline-LD and negative-control resources, file-retention rules, safety margin, and component-level peak-space budget are frozen.

## Provenance and freeze state

All access routes in the source table are official EMBL-EBI, IGSR, NCBI GEO, or UCSC endpoints. Metadata were checked through study records, release tables, directory indexes, manifests, metadata YAML, and HTTP headers only. The five GWAS files, all eQTL association and fine-mapping files, all chromatin matrices and archives, and all LD genotype files remain unopened.

Every row has a portable SHA256 over its metadata fields. These hashes lock this feasibility record only. They are not result-file checksums and do not authorize outcome opening. H27 and H28 remain not tested while the branch is stopped pending locks. No enrichment, fine mapping, colocalization, Mendelian randomization, or mediation test was run.
