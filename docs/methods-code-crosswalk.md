# Methods to code crosswalk

| Manuscript method | Configuration | Implementation | Frozen result |
|---|---|---|---|
| Primary transfer residual | `config/analysis.yaml` | `analyses/primary/run_primary.py` | `analyses/primary/results` |
| Signed vectors and rerouting | `config/analysis.yaml` | `analyses/vectors/run_vector_analysis.py` | `analyses/vectors/results` |
| Network decomposition | `config/network_decomposition.yaml` | `analyses/network_decomposition/run_network_decomposition.py` | `analyses/network_decomposition/results` |
| Context dynamics | `config/context_dynamics.yaml` | `analyses/context_dynamics/run_context_dynamics.py` | `analyses/context_dynamics/results` |
| Guide dose response | `config/guide_dose_response.yaml` | `analyses/guide_dose_response/run_analysis.py` | `analyses/guide_dose_response/results` |
| Untouched RPE1 benchmark | `config/external_benchmark.yaml` | `analyses/external_benchmark/run_benchmark.py` | `analyses/external_benchmark/results` |
| GRN benchmark | `config/external_benchmark.yaml` | `analyses/grn_benchmark/run_beeline.py` | `analyses/grn_benchmark/results` |
| K562 replication | `config/analysis.yaml` | `analyses/replication/run_k562.py` | `analyses/replication/results` |
| Natural genetics | `config/analysis.yaml` | `analyses/natural_genetics/run_eqtlgen.py` | `analyses/natural_genetics/results` |
| Disease convergence | `config/analysis.yaml` | `analyses/disease/run_disease.py` | `analyses/disease/results` |
| Locus integration | `config/causal_triangulation.yaml` | `analyses/loci/build_locus_cards.py` | `analyses/loci/results` |
| Causal gates | `config/causal_triangulation.yaml` | `analyses/causal_triangulation/run_causal_triangulation.py` | `analyses/causal_triangulation/results` |
| Multiverse | `config/multiverse.yaml` | `analyses/multiverse` | `analyses/multiverse/results` |
| Figures | `config/colors.yaml` | `figures` | panel-level `figure_data` directories |
| Figure QC | figure specifications | `analyses/figure_qc/run_figure_qc.py` | `analyses/figure_qc/results` |
