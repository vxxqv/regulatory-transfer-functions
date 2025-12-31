# Methods to code crosswalk

| Manuscript method | Configuration | Implementation | Frozen result |
|---|---|---|---|
| Primary transfer residual | `config/analysis.yaml` | `analyses/primary/run_primary.py` | `analyses/primary/results` |
| Signed vectors and rerouting | `config/analysis.yaml` | `analyses/vectors/run_vector_analysis.py` | `analyses/vectors/results` |
| Network decomposition | `config/network_decomposition.yaml` | `analyses/network_decomposition/run_network_decomposition.py` | `analyses/network_decomposition/results` |
| Context dynamics | `config/context_dynamics.yaml` | `analyses/context_dynamics/run_context_dynamics.py` | `analyses/context_dynamics/results` |
| Guide dose response | `config/guide_dose_response.yaml` | `analyses/guide_dose_response/run_analysis.py` | `analyses/guide_dose_response/results` |
| Original RPE1 falsification benchmark; previously inspected for later extensions | `config/external_benchmark.yaml` | `analyses/external_benchmark/run_benchmark.py` | `analyses/external_benchmark/results` |
| GRN benchmark | `config/external_benchmark.yaml` | `analyses/grn_benchmark/run_beeline.py` | `analyses/grn_benchmark/results` |
| K562 replication | `config/analysis.yaml` | `analyses/replication/run_k562.py` | `analyses/replication/results` |
| Natural genetics corrected audit | `analyses/natural_genetics/audit_correction_01.yaml` | `analyses/natural_genetics/run_eqtlgen_audit.py` | `analyses/natural_genetics/results_audit_corrected` |
| Disease convergence | `config/analysis.yaml` | `analyses/disease/run_disease.py` | `analyses/disease/results` |
| Locus integration | `config/causal_triangulation.yaml` | `analyses/loci/build_locus_cards.py` | `analyses/loci/results` |
| Causal gates | `config/causal_triangulation.yaml` | `analyses/causal_triangulation/run_causal_triangulation.py` | `analyses/causal_triangulation/results` |
| Multiverse | `config/multiverse.yaml` | `analyses/multiverse` | `analyses/multiverse/results` |
| Molecular cascade verification | `config/molecular_cascade.yaml` | `analyses/molecular_cascade/run_analysis.py` | `analyses/molecular_cascade/results` |
| State-response decomposition | `config/state_response_decomposition.yaml` | `analyses/state_response_decomposition/run_analysis.py` | `analyses/state_response_decomposition/results` |
| Information retention | `analyses/information_retention/specification.json` | `analyses/information_retention/run_analysis.py` | `analyses/information_retention/results` |
| Independent confirmation gates | frozen confirmation manifests | `analyses/independent_confirmation/run_genetic_confirmation.py` | `analyses/independent_confirmation/results` |
| Compositionality availability audit | `config/cell_systems_expansion.yaml` | `analyses/compositionality/run_analysis.py` | `analyses/compositionality/results` |
| Three-dimensional transfer landscape | `config/pyvista_transfer_landscape.yaml` | `figures/supplement/S24/make_figure.py` | `figures/supplement/S24` |
| Figures | `config/colors.yaml` | `figures` | panel-level `figure_data` directories |
| Figure QC | figure specifications | `analyses/figure_qc/run_figure_qc.py` | `analyses/figure_qc/results` |
