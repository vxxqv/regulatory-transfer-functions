# Genetic metadata screen

## Scope and freeze

This is a metadata-only screen for H27 and H28. It covers the 14 frozen immune-disease labels and one GWAS Catalog study candidate per label. The source table also records the common LD, immune-cell eQTL, and state-relevant chromatin components, failed alternate chromatin sources, the rejected asthma family-history proxy, and three planned loci excluded because they were inspected before the independent-confirmation freeze.

No GWAS summary-statistic file, eQTL association file, chromatin signal file, peak file, or other outcome file was opened or downloaded. Only official study and release metadata were inspected. No downstream genetics was run.

## Denominator and source bundle

The denominator is 14 disease candidates and 28 source records. The prospective bundle uses GWAS Catalog study and download metadata, the 1000 Genomes NYGC 30x GRCh38 panel with its ancestry-matched subset to be fixed before opening, eQTL Catalogue release 7 T-cell resources QTS000040 and QTS000041, GEO ATAC study GSE118189, and GEO H3K27ac CUT&RUN study GSE244035.

The eQTL Catalogue REST API returned HTTP 410 and is recorded as unavailable. Official release, access, and license pages supplied metadata only. ENCODE experiments ENCSR841LHT, ENCSR991PBP, and ENCSR561KOM are retained as failed alternatives because replication, quality, or paired-state requirements were not met.

For each selected GWAS record, the official download directory and companion metadata YAML were checked without opening the data file. The candidate table records the declared data-file URL, metadata URL, file name, file format, build, HTTP HEAD byte size, provider MD5, and file-metadata variant count when available. None of the companion metadata files reported a variant count, so that field is explicitly unavailable. Catalog-reported variant counts remain separate and do not establish file completeness by themselves.

The common bundle is capable, at the metadata level, of a genome-wide competitive test or partitioned-heritability test with an independent molecular convergence check only for candidates whose phenotype and completeness gates pass. It does not authorize a nominal pathway analysis or any outcome opening.

## Frozen selection rule

The frozen ordering is applied without a threshold change: required metadata gates first, then independent cohort and source separation, then the frozen mapped-target, common-gene, and repeated-unit fields, and finally accession ascending. Those three outcome-derived ranking fields are unavailable and tied for every candidate because outcomes remain unopened. The five eligible candidates are therefore ranked by accession ascending. The rank is selection order only and is not evidence strength.

GCST90476372 is rejected because it measures sibling history of asthma. The replacement, GCST90475259, is the lowest-accession direct European asthma study among the dense imputed direct-asthma candidates passing the outcome-blind metadata gates. Its official metadata names the phenotype `Asthma`, the GWAS-SSF v1.0 file, GRCh38, byte size, and provider MD5.

The generic autoimmune-disease row stops because GCST90012738 reports a coagulation screen rather than the frozen disease phenotype. The inflammatory-bowel-disease loose composite and the broad psoriasis and rheumatoid-arthritis PheCodes also stop on phenotype specificity. Six sparse non-imputed studies have official files but no file-metadata variant count, so their complete genome-wide status is unavailable without opening outcomes. Five rows pass phenotype and completeness screening and advance only to final SHA256 verification and dataset-role locking.

## Gate decisions

- H27 metadata capability: ADVANCE_TO_CHECKSUM_AND_ROLE_LOCK for 5 of 14 candidates.
- H27 phenotype failure: STOP for 4 of 14 candidates.
- H27 completeness unavailable: 6 of 14 candidates, including one row that also stops on phenotype specificity.
- H27 overall unavailable: 5 of 14 candidates after applying phenotype failure first.
- Provider MD5 metadata: listed but not verified for all 14 candidate files.
- Final full-file SHA256 gate: unavailable for all 14 candidates because outcome files remain unopened.
- Outcome-opening gate: unavailable for all 14 candidates. No candidate is open for analysis.
- H27 result: unavailable. Competitive enrichment and partitioned heritability were not run.
- H28 credible-set and colocalization gates: unavailable for all 14 candidates before outcome opening and fine mapping.
- H28 instrument strength, directionality, heterogeneity, pleiotropy, and leave-one-instrument gates: unavailable for all 14 candidates.
- H28 mediation: unavailable for all 14 candidates. No direct-regulation or causal claim is made.

The H27 analysis plan retains primary MHC exclusion and matched-size, expression-matched random-gene, and negative-locus controls. These are plan gates only. They have no result status.

## Checksums

Each row has a provisional metadata SHA256. It is computed over every TSV field before `provisional_metadata_sha256`, in header order, as UTF-8 text with one `field=value` pair per line and LF line endings. This makes verification independent of Windows or Unix file line endings. The provisional hash covers screening metadata and gate decisions. Provider MD5 values are recorded from companion metadata but remain unverified. They do not substitute for the verified full-file SHA256 and locked role required before outcome opening.
