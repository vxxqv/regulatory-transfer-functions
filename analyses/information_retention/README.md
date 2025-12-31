# Information retention

This block retains all 4,399 complete-state targets and 13,197 matched state summaries. It measures conditional cross-state identity decoding, not fully adjusted or independently replicated regulatory information. None of H12-H17 passes its confirmatory availability gate.

## Design

The expansion freeze and the local specification precede these decoder outcomes. Sixteen input hashes are checked against the expansion manifest, allowing only LF versus CRLF checkout equivalence for text while requiring exact binary bytes. Existing transfer values, classes and module loadings are unchanged.

Target identity is a class-known prediction task: each held-out state has two other-state training profiles for every target. The two training states are swapped for inner temperature selection. No held-out profile contributes to nuisance fitting, centroid construction or temperature selection. Unseen-class target-held-out classification would be invalid here. The supplied ten target folds are retained for accounting, not misrepresented as identity-decoder splits.

All 4,399 cohort cis features are removed globally, leaving 5,883 response genes. This stricter mask prevents the position of a target-specific cis zero from encoding the label. Training-only response regressions estimate directions associated with cis magnitude, target expression, cell count, guide count and training-state means. Their orthonormal span is removed from both training and test profiles. Only training-state mean directions are removed; a distinct held-out-state mean is not estimated from its test profiles.

The cosine decoder and the squared-distance centroid decoder operate in this residual gene space. A cosine comparator uses the unchanged 30-module loading basis after the same gene mask. The latter basis was estimated from the full CD4 cohort and is a descriptive comparator, not independently fitted representation validation. The uniform prior is retained. Required generic templates and independent profile units are absent, so confirmatory regularized multinomial fitting is stopped before fitting rather than used to bypass its gate.

For balanced K-class labels, H(Y) + E[log q(Y|X)] is a variational lower bound on information for a fixed decoder. The reported finite-sample score is log(4,399) minus held-out log loss, in nats. It is conditional on this sampled-state task and preprocessing, not total molecular information. Negative individual scores are retained. Subgroup averages retain the same global prior and are performance summaries, not subgroup information bounds. Method: [Barber and Agakov, 2003](https://papers.nips.cc/paper_files/paper/2003/hash/a6ea8471c120fe8cc35a2954c9b9c595-Abstract.html); see also [Poole et al., 2019](https://proceedings.mlr.press/v97/poole19a.html).

Intervals use 1,000 target-bootstrap replicates, conditional on the fitted decoders, without pretending to include donor or model-refitting variability. The partial matched null permutes held-out labels within state, expression quintile and response-degree quintile. It preserves the fixed trained decoder and tests identity alignment; it is not a refitted-decoder null. An independent network-degree annotation is unavailable, so this partial null does not fulfill the full frozen null requirement. Nine one-sided diagnostic tests form one BH family. Three paired state contrasts form a separate two-sided bootstrap family. No confirmatory H12-H17 P values are manufactured.

## Results

RNA-fingerprint scores are 0.861 nats in Rest (95% CI 0.791-0.935), 0.946 at 8 hours (0.870-1.019), and 0.853 at 48 hours (0.778-0.921). Top-1 fractional accuracy is 12.37%, 14.21% and 13.20%, respectively, against a 0.0227% balanced prior. The module comparator scores 0.307, 0.326 and 0.305 nats. Both exceed the partial matched null in each state (BH q = 0.00150). The distance decoder selects inverse temperature zero in all states and provides no information beyond the label prior (P = q = 1).

The 8-hour score exceeds Rest by 0.085 nats (95% CI 0.032-0.134), then decreases by 0.093 nats at 48 hours (0.041-0.146 in the decreasing direction). Both paired contrasts have q = 0.00150. Rest versus 48 hours remains unresolved (difference -0.008, CI -0.066 to 0.050, q = 0.771). These are conditional diagnostics, not confirmation of H15.

Buffered-class scores are small: 0.091, 0.074 and 0.089 nats. Most buffered profiles have no detected response: 362/416, 379/420 and 435/488 in Rest, 8 hours and 48 hours. This does not establish that buffering preserves identity better than magnitude. Amplified and intermediate classes, and every target-level score, remain in the outputs.

Calibration is imperfect. RNA-fingerprint top-label slopes are 0.318, 0.307 and 0.309, with intercepts -0.608, -0.533 and -0.618. Nominal 95% posterior label-set coverage is only 93.36%, 92.33% and 91.95%. Coverage uses randomized boundary ties and is not conformal coverage. No recalibration was fitted to the held-out outcomes.

There are 3,827 zero full-response profiles, increasing to 5,275 after the common gene mask; 434 targets have zero full response in all states. These are zeros in the significance-filtered matrix, not proof of biological silence. No coordinates or response vectors were imputed. Computation took 79.1 seconds and produced 4.96 MiB of analysis outputs. Peak process memory was not recorded because the optional process counter was unavailable; the analysis was not repeated to fill that field.

## Boundaries

Library-size direction, stress/proliferation templates, independent guide/donor response vectors, external signed profiles and independent regulator-family labels are not supplied. Response-derived module labels cannot become independent decoder outcomes. Aggregate guide, donor and K562 correlations do not resolve profile-level information, reliability or measurement error.

Reliable equivalence classes are not estimated from three state summaries. The required 20/30/40-resolution, negative-gene, template-removal and 80% stability checks are not identifiable. H16-H17 retain a deferred input contract. H12-H17 are all unavailable, not negative tests and not positive conclusions. The primary information gate remains false because complete adjustment, guide/donor replication and independent external association are missing.

## Reproduce

Use the existing study environment and the source root containing the unchanged frozen inputs:

```text
python analyses/information_retention/freeze.py --source-root SOURCE_ROOT
python analyses/information_retention/run_analysis.py --source-root SOURCE_ROOT
python figures/supplement/S28/make_figure.py
```

Set `INFORMATION_SOURCE_ROOT` for source-backed tests, then run `pytest tests/test_information_retention.py tests/test_context_dynamics.py -q`. The tests verify all denominators, class-known splits, nested selection, null/FDR families, unavailable gates, source-mask correspondence, 75 independently reconstructed gene-space scores and 225 lower-bound calculations. S28 has four exact source tables, editable SVG, vector PDF and a 4,500-pixel PNG. The vector files contain no raster layers.
