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

For perturbation target `i` in context `c`, the observed distal response is `D_ic`, the number of significant downstream genes after excluding the perturbed target. The proximal input is `C_ic`, the absolute on-target log fold change. The primary empirical transfer residual is the out-of-fold residual from a model of `log(1 + D_ic)` given `C_ic`, baseline target expression, target-cell count, guide count, and context. Cross-validation is grouped by target gene, preventing the same target from entering training and test folds.

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

The protocol, quality filters, transfer outcome, covariates, cross-validation grouping, tail thresholds, and multiplicity families are frozen before inspecting associations with disease labels. Additional models are explicitly marked exploratory.
