# Genetics protocol lock

## Decision

`READY_FOR_CHECKSUM_DOWNLOAD`

The pre-outcome protocol and resource gates are complete, and the locked sequential plan fits the recorded free-space snapshot. This is a scientific readiness decision, not authorization to transfer files. A later execution action may perform sequential checksum downloads in the locked order. Outcome opening and genetics execution remain prohibited until every selected file has passed its provider checksum when one exists, a local SHA256 has been recorded, the checksum manifest has itself been hashed, and the immutable role order has been reconfirmed. No association or molecular outcome file was opened for this lock.

## Scope and stopping rule

The lock covers five immune diseases, two selected falsification diseases, European LD, state-resolved CD4 eQTL resources, ATAC-seq, H3K27ac, and fixed gene and genome annotations. Each branch must add an independent evidence type or resolve a central uncertainty, and it must be capable of changing or narrowing a conclusion. A favorable result is not required. A branch stops without a replacement when availability, checksum, schema, power, novelty, or results-reporting gates fail.

## Baseline LD

The primary S-LDSC reference is Broad Institute baseline-LD v2.2 on GRCh37 for chromosomes 1 through 22. The exact annotation, frequency, PLINK, HapMap3 no-MHC weight, regression-SNP, and version files are fixed in the configuration and resource table with canonical URLs and provider MD5 values. Primary S-LDSC uses HapMap3 SNPs outside the MHC, overlapping annotations, printed coefficients, and delete-values. The GRCh38 1000 Genomes NYGC callset is a separate fine-mapping LD resource and is not a substitute for baseline-LD.

## Disease denominator and falsification controls

The positive denominator is asthma, Hashimoto thyroiditis, multiple sclerosis, type 1 diabetes, and ulcerative colitis. Binary-trait effective sample size is fixed as `4/(1/cases+1/controls)`, giving a positive range of 10,980.26 to 97,623.66.

The negative screen was restricted before outcomes to direct clinical disease PheCodes from the same MVP European publication and dense 44.3-million-variant GWAS-SSF release. Family-history traits, biomarkers, medication traits, symptom-only traits, broad proxies, and primary autoimmune, infectious, or neoplastic diseases were ineligible. The permitted effective-sample-size interval is 0.90 times the smallest through 1.20 times the largest positive value, or 9,882.23 to 117,148.39. Within the fixed structural-hernia family, candidates were ranked by absolute log effective-sample-size distance to the largest positive study. Within the fixed gastric-motility family, candidates were ranked against the smallest positive study. Ties are broken by ascending accession. No association statistic enters the screen.

All five screened candidates remain in the resource table. Abdominal hernia was outside the sample-size interval. Ventral and umbilical hernia were eligible but ranked behind inguinal hernia. Inguinal hernia, with 30,294 cases, 410,088 controls, and effective N 112,840.27, was selected as the structural control. Gastroparesis, with 2,813 cases, 446,383 controls, and effective N 11,181.54, was the sole eligible fixed gastric-motility candidate and was selected as the low-power control.

Hernia can involve connective-tissue biology, and gastroparesis can coexist with inflammatory or autoimmune disease. They are therefore not asserted to be biologically immune-free. They are acceptable specificity controls because the recorded phenotypes are direct non-immune diagnoses, their cohort, ancestry, provider, publication, imputation scale, and file schema match the MVP immune studies, and their effective sample sizes bracket the immune range. Their role is falsification, not proof of mechanistic absence. Enrichment in either selected control fails the specificity gate and is reported without choosing a replacement.

A positive result in either negative disease does not stop or alter the five locked immune-disease analyses. Every locked disease that passes its checksum, resource, and post-opening schema gates is analyzed and reported regardless of results observed earlier in the opening order. Negative-control enrichment fails only the specificity hypothesis. Only a checksum failure, schema failure, or unavailable locked resource stops the affected disease pipeline.

## CD4 program freeze

Five existing CD4 annotation sources are locked by path, byte count, and SHA256 before any disease outcome opens: transfer phenotypes, reference module loadings, regulatory edge evidence, core and state-residual decomposition, and context-switch scores. Their sorted path-and-hash manifest SHA256 is `54579caf9c69001aa2f70edf352f59b8eb1235f537231a698c580afbc0224ef8`. Program definitions may not use disease or locus fields and may not be rebuilt after a disease outcome opens. These files are already present project outputs; recording their hashes did not require a new outcome download.

## Matched negative loci and regions

Ten control windows are selected per locus without replacement within a disease. Chromosome is exact. Fixed calipers are MAF difference at most 0.02, locus-length and mean-LD-score ratios from 0.80 to 1.25, gene-density difference at most 2 genes per Mb, log10 distance-to-TSS difference at most 0.25, accessibility-fraction difference at most 0.05, and mean 100-bp Umap mappability difference at most 0.05. Controls must be outside the MHC and blacklists and at least 2 Mb from every index locus. Nearest neighbors minimize standardized Manhattan distance, with chromosome then start coordinate as the tie-break. Association values and significance are forbidden matching variables. Fewer than five controls makes that locus comparison unavailable; calipers are not relaxed.

## Build harmonization

GSE118189 ATAC intervals are BED 0-based half-open hg19 coordinates. Forward conversion uses the pinned UCSC hg19-to-GRCh38 chain with `minMatch=0.95` and multiple mapping disabled. An interval is retained only if it has exactly one same-chromosome canonical output, retains at least 95% of its length, and maps back through the reciprocal chain to exactly one same-chromosome interval overlapping at least 95% of the source length. Split, multi-map, chromosome-change, alternate-contig, short, and reciprocal failures are rejected. The pinned hg19 blacklist is applied before mapping and the hg38 blacklist after mapping. A fixed 0.99 sensitivity is run without replacing the primary rule. Counts and base-pair denominators are reported for every rejection class.

## Opening order and schema gates

The five existing CD4 program-annotation paths, byte counts, individual hashes, and combined manifest hash are verified first. References are then acquired and verified before outcomes. The two negative diseases open first, followed by the five immune diseases, the three state eQTL pairs, ATAC, and H3K27ac. QTS000040 remains excluded because its subtype role and state mapping were unresolved before outcomes.

After checksum verification, each GWAS must contain chromosome, position, effect and other alleles, exactly one of beta or odds ratio, standard error, P value, sample size, frequency, and INFO for imputed variants. Values must meet the fixed allele, range, duplication, coverage, and harmonization rules. Missing required fields, truncation, a missing chromosome, excessive duplicates, more than 5% unresolved effect directions, or less than 90% expected dense coverage stops that disease. A locus stops below 100 shared variants or 80% shared GWAS, eQTL, or LD coverage. There are no resource substitutions.

## Analysis order

CD4 program annotations are frozen before the first negative GWAS. The complete negative-disease pipelines then precede immune outcome inspection, but their results cannot stop a schema-eligible locked disease. Index loci are derived only after a GWAS opens, using the fixed genome-wide significance, clumping, and boundary rules. Matched negative loci are then derived with the already frozen outcome-independent covariates and calipers. Competitive MAGMA analysis precedes S-LDSC, SuSiE RSS fine mapping, coloc-SuSiE, molecular convergence, instrument gates, and gated mediation. The MHC-included analysis is a sensitivity. Matched controls and rewiring are evaluated before Benjamini-Hochberg correction and the complete denominator. Method versions, priors, windows, thresholds, covariates, and causal gates are fixed in the configuration. No outcome-tunable alternative is permitted.

## Retention and capacity

Free space was 182,042,304,512 bytes at 2026-09-13 21:16:03 UTC. Locked sources require 56,937,355,656 bytes. Locked summaries reserve 8,000,000,000 bytes, the largest temporary stage reserves 36,000,000,000 bytes, and tool caches reserve 2,000,000,000 bytes. The peak before safety is 102,937,355,656 bytes. Adding a 25% free-space safety margin of 45,510,576,128 bytes gives 148,447,931,784 bytes, leaving 33,594,372,728 bytes.

LD and analysis work proceeds chromosome by chromosome. Original provider files are never deleted. A temporary intermediate may be retired only after its reproducible command or specification, source hashes, locked summary, summary hash, and numerical checks are recorded. Execution stops if current free space falls below the fixed safety margin.

## Locked interpretation

The protocol is scientifically ready for a later sequential checksum-download action, not for outcome inspection. Any post-opening schema failure is a prespecified stop, not permission to substitute a new dataset. Negative-control results cannot truncate the locked disease denominator. Supported, null, contradictory, unavailable, and underpowered diseases and loci remain in the denominator.
