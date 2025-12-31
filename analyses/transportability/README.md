# Transportability frontier

This block evaluates H7-H11 under the frozen extended validation rules. CD4 is the development source. K562 and RPE1 were inspected before the validation freeze, and no new untouched external outcome dataset is available.

The stored external outputs do not retain target-level guide, replicate, or split-half reliability for K562 or RPE1. Reliability-corrected concordance is therefore not point identified. The analysis reports bounded partial-identification regions under the frozen 0.10 to 1.00 reliability limits and does not divide an observed correlation by a single estimated reliability.

K562 supports an observed-concordance diagnostic for rerouting and molecular evidence. It does not support H8 or H11 after reliability correction. RPE1 supports a developmental risk-coverage diagnostic using its stored out-of-fold predictions, baseline control expression, and CD4-only source features. RPE1 perturbed-cell cis and distal outcomes are excluded from applicability features. RPE1 is not treated as an untouched holdout or a held-out system. The three-compatible-system gate and confirmatory external gate remain binding.

All 5,640 K562 target-state comparisons and all 490 RPE1 targets are retained. Missing features, failed gates, nulls, intervals, calibration, coverage, costs, and hypothesis decisions are machine readable in `results/`.

Run `python analyses/transportability/run_analysis.py`, then `python figures/supplement/S29/make_figure.py` and `pytest tests/test_transportability.py -q` in the existing study environment.
