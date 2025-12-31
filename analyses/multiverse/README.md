# Specification-Curve and Multiverse Robustness Analysis

`config/multiverse.yaml` freezes the complete post-primary sensitivity grid before execution. The analysis covers seven central result families and retains every valid specification, including unfavorable, null, and underpowered estimates.

The first completed implementation run exposed two estimands whose signs were fixed by definition: the lower-tail buffering residual and the uncentered context range. Those scalar outputs were invalidated before interpretation, figure use, or manuscript use. `method_amendment_01.json` records the replacement held-state buffering statistic and permutation-centered context statistic; `amendment_01_manifest.json` shows that the revised estimands were frozen before the corrected rerun. This is an audited implementation amendment within a retrospective sensitivity analysis, not a preregistration.

Outputs comprise the complete specification table, family summaries, deletion and influence diagnostics, negative-control results, unavailable-unit audit, compute log, and the full specification-curve figure with an external legend. Applicability rules prevent irrelevant dimensions from creating duplicate rows.
