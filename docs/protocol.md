# Frozen Analysis Protocol

## Objective

Estimate how a proximal regulatory perturbation is transformed into a distal transcriptional response in human immune cells. The confirmatory analysis asks whether perturbations with comparable on-target effects differ reproducibly in trans-response magnitude, network destination, and context dependence.

## Evidence hierarchy

1. Primary causal evidence: genome-scale CRISPR interference and Perturb-seq in primary human CD4 T cells.
2. Orthogonal causal evidence: noncoding CRISPR interference, variant editing, and arrayed perturbation studies.
3. Natural-genetic evidence: trans-eQTL and context-specific eQTL resources.
4. Disease evidence: fine-mapped immune-trait loci and matched enrichment tests.
5. Mechanistic support: chromatin state, enhancer-gene links, three-dimensional contacts, transcription-factor programs, and protein interactions.

## Primary estimand

For perturbation target `i` in context `c`, the observed distal response is `D_ic`, the number of significant downstream genes after excluding the perturbed target. The proximal input is `C_ic`, the absolute on-target differential-expression z-score released by the source study as `ontarget_effect_size`. The primary empirical transfer residual is the out-of-fold residual from a model of `log(1 + D_ic)` given `C_ic`, baseline target expression, target-cell count, guide count, and context. Cross-validation is grouped by target gene, preventing the same target from entering training and test folds.

Positive residuals indicate more distal response than expected from perturbation strength and observability. Negative residuals indicate less distal response than expected. The terms amplified and buffered are reserved for the upper and lower prespecified deciles after quality control. These labels describe comparative transfer phenotypes, not biochemical proof of active amplification or buffering.

## Inclusion criteria

- Significant on-target knockdown at the source study threshold.
- Target not flagged for low expression.
- No neighboring-gene knockdown flag.
- No distal predicted off-target flag.
- At least two guides.
- At least 200 target cells.

## Context switching

The primary switch statistic is the within-target range of cross-fitted transfer residuals across Rest, Stim8hr, and Stim48hr. A secondary rerouting statistic uses the entropy of condition-specific perturbation-cluster assignments. Targets must have data in all three conditions. State effects are estimated with target-level resampling and multiplicity control within the context-switching family.

## Network decomposition

Downstream programs are represented by the source study's perturbation clusters and downstream-gene summaries. Module contributions are calculated without disease labels. Disease annotations are joined only after transfer phenotypes and network modules are frozen.

## Validation

- Guide-level and donor-pair reproducibility, where available.
- Independent arrayed bulk RNA-seq and polarization experiments.
- Cross-system comparison with K562 perturbations.
- Natural trans-eQTL concordance.
- Independent CRE perturbation resources.
- Matched disease-locus enrichment.

## External benchmark and falsification challenge

All CD4-derived outcomes, features, labels, module loadings, model predictions, and target-fold assignments are locked by SHA-256 digest before processing the falsification holdout. Replogle K562 results inspected during development remain a labeled external replication analysis and are not represented as untouched. The confirmatory falsification holdout is a separately acquired Replogle RPE1 CRISPRi Perturb-seq screen. Its perturbation outcomes cannot alter the CD4 feature definitions, model hyperparameters, endpoints, eligibility rules, folds, null procedures, or decision thresholds.

The primary external endpoint is `log(1 + number of differentially expressed genes)` for each eligible shared perturbation target. Secondary endpoints are the pseudobulk response norm and cross-system response-vector concordance. Every eligible target is retained in the target-level results table. Targets that fail minimum-cell, gene-overlap, or effect-estimation requirements remain in an exclusions table with one explicit reason.

All scalar models use the same deterministic target-held-out folds. Comparators are a global mean, simple observability covariates, regularized linear regression, a prespecified nonlinear model, standard network propagation, and a graph-based readout. The full transfer model adds only frozen CD4 transfer features. Component ablations remove proximal magnitude, scalar transfer residual, transfer class, context summaries, module scores, or network features one block at a time. Hyperparameters are fixed from the CD4 analysis or selected only within each training fold without access to held-out targets.

Inference uses target-bootstrap confidence intervals, matched target-label permutations within expression and response-degree strata, and degree-preserving network rewiring. Evaluation includes out-of-fold R-squared, mean absolute error, Spearman correlation, calibration slope and intercept, 50%, 80%, and 95% interval coverage, interval width, wall time, and peak memory. Confidence intervals and null tests resample targets, not rows. All model and ablation results are reported, including negative and underpowered comparisons.

Hypothesis decisions are mechanical. A hypothesis passes only when its prespecified effect has the expected sign and its target-bootstrap 95% interval excludes the null after correction within the external-benchmark family. It fails when the interval excludes the prespecified minimally relevant effect in the unfavorable direction or a calibrated null test rejects in the unfavorable direction. Otherwise it remains unresolved. No endpoint may substitute for a failed primary endpoint.

As a separate real-data network-recovery stress test, the workflow evaluates frozen inference procedures on the experimental BEELINE single-cell datasets with their matched cell-type ChIP-seq reference networks. Gold-standard edges are used only for scoring. Dataset selection, gene filtering, candidate edges, metrics, and randomization rules are fixed before scores are inspected.

## Specification-curve and multiverse robustness analysis

A complete post-primary robustness multiverse is locked in `config/multiverse.yaml` before any multiverse estimates are calculated. This timing is stated explicitly: the primary results existed before the multiverse was specified, so the multiverse is a structured sensitivity analysis and not a prospective preregistration. Its grid, seeds, endpoint-specific applicability rules, and decision thresholds cannot be changed after execution begins.

The seven central result families are transfer gain, buffering, rerouting, context switching, cross-system conservation, natural-genetic concordance, and disease convergence. Each is re-estimated across every applicable combination of the frozen preprocessing thresholds, transfer definitions, model families, covariate sets, tail cutoffs, module resolutions, and cross-validation seeds. Non-applicable dimensions are recorded explicitly rather than multiplied into duplicate specifications. The complete specification table includes every estimate, standard error, clustered-bootstrap interval, raw P value, adjusted Q value, direction, support status, sample size, and data-exclusion count.

Deletion analyses operate at the highest identifiable experimental unit. They include leave-one-donor, leave-one-guide, leave-one-state, leave-one-target, and leave-one-locus calculations when those identifiers and repeated observations exist. An unavailable deletion analysis is recorded with the absent field and an unresolved status. Influence diagnostics include deleted-estimate change, externally studentized residual or endpoint-appropriate analogue, leverage, Cook's distance where a fitted regression is used, and the maximum share of the aggregate estimate attributable to one independent unit.

Matched permutations preserve prespecified expression, cell-count, response-degree, state, and locus-size strata as applicable. Negative-control exposures include permuted transfer labels, sign-shuffled transfer scores, degree-preserving network rewiring, and random modules matched on size. Negative-control outcomes include random-gene response sets matched on size and detectability, chromosome-shifted trans pairs, and phenotype labels permuted within locus-size strata. Confidence intervals resample target, donor, or locus clusters as appropriate, and false-discovery correction is performed separately within each central-result family.

For each family, robustness is summarized as the proportion of valid specifications with the expected direction, the proportion supported at Q below 0.05, the median and full range of estimates, and the worst-case deletion result. A conclusion is labeled broad when at least 80% of valid specifications have the expected direction and at least 50% remain supported after correction, mixed when direction stability is at least 60% but corrected support is below 50%, narrow when fewer than 60% share the expected direction or support is confined to one setting of a dimension, and unresolved when power or required identifiers are insufficient. The complete curve is displayed; favorable specifications are not filtered.

## Orthogonal causal triangulation and mediation

The causal-triangulation denominator is every locus in the frozen deep-locus panel that has a statistical credible set, an Open Targets locus-to-gene candidate measured in the primary CD4 screen, and an immune-mediated disease association. Eligibility is evaluated without reference to whether downstream evidence is favorable. The initial denominator is therefore three loci: GATA3-asthma, STAT3-inflammatory bowel disease, and PTPN22-rheumatoid arthritis. Any later expansion must be labeled exploratory and must report both the original and expanded denominators.

Each locus is evaluated against distinct evidence gates: statistical fine-mapping; GWAS and cis-eQTL colocalization; credible-set overlap with active CD4 chromatin; enhancer-to-gene evidence; significant perturbational cis effect; a signed downstream perturbation program; and concordant trans-eQTL direction. Evidence absence is not evidence of convergence. A causal-chain designation requires every gate to pass without a material directional contradiction.

Colocalization requires a shared-variant posterior probability of at least 0.80, or an explicitly comparable source-specific threshold, and adequate regional variant coverage. Enhancer-to-gene support requires either experimental perturbation evidence or at least two concordant orthogonal links among activity-by-contact, promoter capture Hi-C, co-accessibility, or curated locus-to-gene evidence. Chromatin overlap must use a 95% credible-set variant and a CD4-relevant accessible or H3K27ac-marked element. Perturbational cis and downstream evidence must use the frozen CD4 estimands and signed programs.

Two-step mediation is attempted only when independent or conditionally independent instruments are available for both the variant-to-cis and cis-to-distal steps, alleles can be harmonized, colocalization passes, and instrument strength is adequate. The primary estimand is the product of the two step-specific effects. Sensitivity analyses test heterogeneity, horizontal pleiotropy, Steiger directionality, leave-one-instrument influence, alternate candidate genes, and matched negative loci. Colocalization-aware Mendelian randomization must condition interpretation on the probability of a shared causal variant.

No mediation claim is made if colocalization fails, if fewer than the prespecified number of valid instruments remain, if allele harmonization is ambiguous, if horizontal pleiotropy or heterogeneity invalidates the model, or if directionality favors the reverse path. Such loci are retained and graded unresolved or unsupported. Evidence tiers, full gate results, instrument exclusions, negative controls, and the complete locus denominator are reported in `config/causal_triangulation.yaml` outputs.

## Study-wide regulatory verification and molecular cascades

The regulatory verification is a retrospective prespecified extension. Primary transfer outcomes had already been estimated, but the eligible transcription factors, target states, time labels, resource versions, genome-build rules, evidence thresholds, null models, exclusion rules, and reporting denominator are locked before regulatory resources are joined to those outcomes. This distinction prevents the extension from being represented as a prospective preregistration.

The eligible edge denominator comprises every nonzero edge in the frozen significant CD4 response matrix for a quality-controlled perturbed target represented in at least one compatible curated or motif resource. Zero-response target-state combinations remain in the target denominator. Regulatory resources include human JASPAR-labelled target sets, DoRothEA, RegNetwork, TRRUST, CollecTRI, primary CD4 ATAC-seq and H3K27ac, CD4 promoter-capture Hi-C, and the only released state-compatible primary CD4 TF occupancy experiment identified in the frozen search, CTCF ChIP-seq in resting cells. TFTG and state-matched TF occupancy outside resting CTCF are recorded as unavailable. Database overlap is corroborating provenance and is not counted as independent replication.

All promoter-capture Hi-C coordinates are lifted from GRCh37 to GRCh38 with the frozen UCSC chain. Both interval endpoints must map to the same chromosome. The primary promoter window extends 2 kb upstream and 500 bp downstream of the strand-aware transcription start site. Primary physical links require a CHiCAGO score above 5 in the mapped CD4 state. Primary active chromatin requires both ATAC and H3K27ac support. Alternative promoter windows, contact thresholds, and an ATAC-or-H3K27ac definition are sensitivity analyses.

Signed curated edges are tested against the expectation that TF knockdown reverses the curated activation or repression direction. A higher-order path requires an affected intermediate TF and a signed curated edge from that intermediate to the distal gene. Edge classes are enhancer-linked, promoter-bound, motif-supported, one-step TF cascade, higher-order cascade, or unsupported indirect. The strongest evidence tier requires compatible TF occupancy, active chromatin, a physical or independently validated enhancer link, a signed CRISPR response, independent guide concordance, and no directional contradiction. Direct-regulation language is prohibited when this gate fails.

Motif tests use controls matched within state and chromosome class on baseline expression, promoter ATAC and H3K27ac peak counts, gene GC content, gene interval length, regulator-target distance, and curated network degree. Inference uses target-clustered bootstrap intervals, matched permutations, degree-preserving network rewiring, false-discovery correction, leave-one-factor, leave-one-state, leave-one-dataset, and target-held-out checks. TF activity uses a consensus of signed weighted-mean and univariate linear-model scores from high-confidence regulons, with broader regulons restricted to sensitivity analysis.

Guide concordance is reported for cis knockdown, full-vector and significant-union correlations, significant-union sign agreement, and significant-gene counts. Guide-resolved signed vectors, module activity, transfer gain, rerouting, disease convergence, and joint donor-by-guide effects are unavailable in the public release and are not reconstructed. Two-guide target-state combinations cannot identify target-specific nonlinear dose curves or cis mediation. They remain underpowered for those estimands.

Timing comparisons use the Rest, 8-hour, and 48-hour cross-sectional states to test direct-edge support, cascade depth, signed propagation, module coherence, and rerouting. They are not interpreted as longitudinal causation. K562 provides direction-aware external replication. RPE1 remains a scalar untouched falsification endpoint because compatible signed target-gene vectors were not retained in the frozen benchmark output. Every supported, contradictory, null, heterogeneous, unavailable, and underpowered target remains in machine-readable tables.

## Negative controls

- Target-label permutations within expression and cell-count strata.
- Condition permutations within target where exchangeability is justified.
- Degree-preserving network rewiring.
- Sign shuffling for signed analyses.
- Nearest-gene and random-program baselines.
- Cross-mappability exclusion for trans-genetic tests.

## Multiplicity and uncertainty

Benjamini-Hochberg correction is applied separately to the prespecified families in `config/analysis.yaml`. Resampling uses the highest independent unit available, normally target gene or donor. Effect sizes and confidence intervals are reported with adjusted P values.

## Confirmatory boundary

The protocol, quality filters, transfer outcome, covariates, cross-validation grouping, tail thresholds, multiplicity families, and external-challenge decision rules are frozen before inspecting the confirmatory labels to which they apply. Analyses whose outcomes had already been inspected are labeled external replication, not untouched confirmation. Additional models are explicitly marked exploratory.
