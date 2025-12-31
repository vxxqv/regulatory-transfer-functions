# State-response decomposition

The frozen extension includes every target measured once in Rest, Stim8hr and Stim48hr: 4,399 targets and 13,197 observations. It reuses the significant signed distal response matrix, normalized by max(abs(cis effect), 0.1), with the cis gene removed. This is a decomposition of detected effects, not a model of unobserved expression or cell trajectories.

For target t, the mean component is c = (x_rest + x_8h + x_48h) / 3. The residual in state s is x_s - c. Total squared response energy equals 3||c||² + sum_s ||x_s - c||². This identity holds across states; individual state energies require the retained core-residual cross term. No trained shared basis is needed for the primary identity.

An arithmetic mean can contain state-specific genes. Conservation is therefore assessed separately through same-sign responses in all three states, using the smallest absolute state response per gene, and prediction of a held-out state from the other two states. Zero significant-response vectors are retained. A zero vector is not an observed absence of biological response.

## Frozen checks

The configuration, target folds and 14 source hashes were frozen before decomposition outcomes. Alternative 10-, 20- and 30-component uncentered SVD bases are fitted only to the four training folds; the fifth fold is projected without refitting. The same target never appears in training and testing. Previously computed module scores are a descriptive comparison only.

The matched null permutes stimulated target identities within the 25 strata defined by quintiles of mean response degree and cis strength. It preserves the marginal state distributions and matches these covariates approximately within strata. There are 999 permutations and 1,000 target-bootstrap replicates. Associations use two-sided comparisons against their matched-null median, with BH correction across six outcomes. Intervals for excess correlation subtract the fixed permutation median from the raw bootstrap interval. They do not include uncertainty in that median. Added predictive value uses the frozen target folds, a fixed ridge penalty of 10, and training-only imputation and scaling. Paired target-bootstrap intervals and centered-bootstrap tests are corrected across three prediction comparisons.

Leave-one-state predictions, all target influences and fixed component sensitivities are supplied. Individual targets have only three condition summaries; population bootstrap intervals do not provide target-specific replication intervals. Guide and donor summaries do not permit component-specific measurement-error, guide-quality or donor-effect models. Those gates remain unavailable here.

## Findings

Of 4,399 targets, 3,965 have nonzero significant-response energy and 434 have none. The mean component accounts for 43.25% of energy on average (95% CI 42.80-43.73%), above the matched-target mean of 33.45% (P = 0.001). Same-sign three-state consensus accounts for 6.01% (95% CI 5.61-6.42%); 1,464 targets have at least one consensus gene. The distinction limits any claim of broad biological conservation.

Held-out state cosine similarity averages 0.278 in Rest, 0.297 at 8 hours and 0.277 at 48 hours among evaluable vectors. Projected core fractions remain near 0.43 across the three frozen resolutions. These subspaces capture only 2.9%, 5.5% and 7.3% of total response energy on average, so agreement is not proof that low-rank components explain the full response.

Raw core-fraction associations with guide, donor and K562 concordance do not exceed their matched null after FDR correction. Adding core fraction to covariates produces no supported held-out prediction improvement: delta R² is -0.0005 for guides, 0.0066 for donors and 0.0007 for K562, with all intervals crossing zero. Core fraction is negatively associated with module rerouting. Motif-score range has a positive raw association but a smaller value than its matched expectation; reporting only the raw sign would be misleading. These two associations are response-derived, not independent molecular evidence. The RPE1 error-penalty association remains unresolved.

The external files contain whole-response concordance or model-prediction summaries, not mapped component-specific signed vectors. The supplied Th1/Th2 comparison contains eight targets, below the frozen 20-target minimum. No new independent T-cell dataset was introduced in this block. Chromatin, enhancer, cascade and direct external core-vector validation require other evidence and remain unavailable here.

## Reproduce

Run from the repository with the existing environment and provide the source root containing the unchanged frozen inputs:

```text
python analyses/state_response_decomposition/freeze.py --source-root SOURCE_ROOT
python analyses/state_response_decomposition/run_analysis.py --source-root SOURCE_ROOT
python figures/supplement/S26/make_figure.py
```

Do not regenerate the freeze when source hashes differ without documenting a new specification. Configuration and canonical cohort hashes use UTF-8 with LF newlines for portability; the original cohort byte hash is also retained. `finalize_statistics.py` reconciles the initial stored outputs after numerical review; it corrects matched-null direction and supplies prediction FDR without refitting models. The primary script now incorporates that reconciliation.

Set `STATE_DECOMPOSITION_SOURCE_ROOT` for source-backed tests, then run `pytest tests/test_state_response_decomposition.py tests/test_context_dynamics.py -q`. Tests independently reconstruct 25 deterministic targets from signed gene values, all six raw correlations and the held-out loss comparisons. Figure S26 has six exact source tables and editable vector exports.

Method references: [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) and [Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html). Implementation versions are inherited from the study environment.
