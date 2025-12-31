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
