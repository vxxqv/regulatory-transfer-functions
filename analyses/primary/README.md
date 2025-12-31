# Primary analysis

`run_primary.py` applies the frozen quality filters, estimates cross-fitted empirical transfer residuals, quantifies state switching, joins perturbation-cluster assignments, and runs the prespecified condition-stratified permutation control.

Example:

```text
python analyses/primary/run_primary.py --source-root path/to/GWT_perturbseq_analysis_2025
```

The input repository must resolve to commit `aa5c84a973c0e1a090b0072dc5b080bf7fbbed38`. Output files are deterministic under the seed in `config/analysis.yaml`.
