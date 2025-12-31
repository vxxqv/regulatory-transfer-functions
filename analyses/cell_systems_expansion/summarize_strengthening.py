from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/cell_systems_expansion/results"
TABLE = ROOT / "tables/table_s9_strengthening_results.tsv"


def row(**values: object) -> dict[str, object]:
    return values


def main() -> None:
    transfer = pd.read_csv(ROOT / "analyses/primary/results/transfer_stability_summary.csv").set_index("metric")
    transfer_sensitivity = pd.read_csv(ROOT / "analyses/primary/results/transfer_measurement_error_sensitivity.csv")
    primary = pd.read_json(ROOT / "analyses/primary/results/primary_results.json", typ="series")
    scalar = pd.read_csv(ROOT / "analyses/multiverse/results/scalar_family_summary.csv").set_index("result_family")
    vectors = pd.read_json(ROOT / "analyses/vectors/results/vector_results.json", typ="series")
    rerouting = pd.read_csv(ROOT / "analyses/vectors/results/composition_sensitivity_summary.csv").set_index("scope")
    guide = pd.read_csv(ROOT / "analyses/guide_concordance/results/hypothesis_tests.csv")
    guide = guide.loc[guide["analysis"].eq("paired_cis_count:target_controlled")].iloc[0]
    state = pd.read_csv(ROOT / "analyses/state_response_decomposition/results/summary_estimates.tsv", sep="\t").set_index("metric")
    state_prediction = pd.read_csv(ROOT / "analyses/state_response_decomposition/results/heldout_prediction_comparison.tsv", sep="\t")
    external = pd.read_csv(ROOT / "analyses/external_benchmark/results/hypothesis_decisions.csv").iloc[0]
    external_models = pd.read_csv(ROOT / "analyses/external_benchmark/results/model_comparison.csv").set_index("model")
    grn = pd.read_csv(ROOT / "analyses/grn_benchmark/results/dataset_heterogeneity.csv").set_index("metric")
    grn_contrasts = pd.read_csv(ROOT / "analyses/grn_benchmark/results/paired_method_contrasts.csv")
    molecular = pd.read_csv(ROOT / "analyses/molecular_cascade/results/hypothesis_decisions.csv")
    molecular_gain = molecular.loc[molecular["family"].eq("transfer_gain")].iloc[0]
    molecular_prediction = pd.read_csv(ROOT / "analyses/molecular_cascade/results/paired_prediction_comparison.csv")
    natural = pd.read_csv(ROOT / "analyses/natural_genetics/results_audit_corrected/hypothesis_tests.csv")
    natural = natural.loc[(natural["subset"].eq("primary")) & natural["scope"].eq("overall") & natural["metric"].eq("direction_agreement")].iloc[0]
    disease = pd.read_json(ROOT / "analyses/disease/results/disease_results.json", typ="series")
    loci = pd.read_csv(ROOT / "analyses/causal_triangulation/results/locus_grades.csv")
    validation = pd.read_csv(ROOT / "analyses/multiverse/results/validation_family_summary.csv").set_index("result_family")

    core = state.loc["core_fraction"]
    reroute = rerouting.loc["overall"]
    grn_auroc = grn.loc["auroc"]
    transfer_wins = int(grn_contrasts["corrected_result"].eq("finite_horizon_transfer_superior").sum())
    simple_wins = int(grn_contrasts["corrected_result"].eq("frozen_simple_comparator_superior").sum())
    core_ranges = state_prediction.set_index("outcome")["delta_r2"].to_dict()
    molecular_ranges = molecular_prediction.set_index("dataset")["delta_r2"].to_dict()

    records = [
        row(
            result_family="Transfer gain and buffering",
            extension_action="Hierarchical shrinkage and class stability",
            original_estimate=f"out of fold R2 {primary['out_of_fold_r2']:.4f}; gain support {scalar.loc['transfer_gain', 'q_supported_fraction']:.3f}; buffering support {scalar.loc['buffering', 'q_supported_fraction']:.3f}",
            original_denominator="15807 target state observations from 6105 targets",
            strengthened_estimate=f"rank rho {transfer.loc['spearman', 'estimate']:.4f}; buffered retention {transfer.loc['buffered_rank_retention', 'estimate']:.3f}; amplified retention {transfer.loc['amplified_rank_retention', 'estimate']:.3f}",
            strengthened_denominator="6105 targets with 611 targets in each original tail",
            confidence_interval=f"rho {transfer.loc['spearman', 'ci_low']:.4f} to {transfer.loc['spearman', 'ci_high']:.4f}; buffered {transfer.loc['buffered_rank_retention', 'ci_low']:.3f} to {transfer.loc['buffered_rank_retention', 'ci_high']:.3f}; amplified {transfer.loc['amplified_rank_retention', 'ci_low']:.3f} to {transfer.loc['amplified_rank_retention', 'ci_high']:.3f}",
            corrected_significance=f"tail retention q {transfer.loc['buffered_rank_retention', 'q_value']:.6f} and {transfer.loc['amplified_rank_retention', 'q_value']:.6f}",
            effect_size_interpretation="Target ordering is highly stable after partial pooling, but only about one third of tail labels remain beyond the original fixed cutoffs after shrinkage.",
            sensitivity_and_influence=f"Variance multipliers 1.0 to 1.5 retain rho at least {transfer_sensitivity['spearman'].min():.3f} and rank retention at least {min(transfer_sensitivity['buffered_rank_retention'].min(), transfer_sensitivity['amplified_rank_retention'].min()):.3f}; all target influence values are retained.",
            evidence_gained="Improved precision and explicit separation of rank stability from threshold dependent labels.",
            remaining_limitation="Target state sampling variances are unavailable, so measurement error is bounded rather than point identified.",
            final_status="mixed",
            materially_improves_manuscript=True,
        ),
        row(
            result_family="Signed programme rerouting and context dependence",
            extension_action="Composition aware matched sensitivity",
            original_estimate=f"median module Jensen Shannon divergence {vectors['median_module_js_divergence']:.4f}; 180 of 180 rerouting specifications supported",
            original_denominator=f"{int(vectors['evaluable_state_pairs'])} evaluable pairs across 3696 targets",
            strengthened_estimate=f"observed minus matched null normalized Aitchison distance {reroute['observed_minus_null_median']:.4f}",
            strengthened_denominator=f"{int(reroute['pairs'])} pairs from {int(reroute['targets'])} targets",
            confidence_interval=f"{reroute['cluster_bootstrap_ci_low']:.4f} to {reroute['cluster_bootstrap_ci_high']:.4f}",
            corrected_significance=f"directional upper tail q {reroute['q_value_upper_bh']:.3f}; lower tail p {reroute['permutation_p_lower']:.6f}",
            effect_size_interpretation="Within target state compositions differ, but remain closer than matched cross target compositions after conditioning on magnitude.",
            sensitivity_and_influence="The opposite direction holds for every state pair; existing module resolution, seed and state deletion sensitivities remain unchanged.",
            evidence_gained="Added a log ratio geometry and a magnitude matched falsification null.",
            remaining_limitation="Guide and donor resolved module compositions are unavailable and the result is conditional on the frozen 30 component basis.",
            final_status="mixed",
            materially_improves_manuscript=True,
        ),
        row(
            result_family="Guide level dose response and concordance",
            extension_action="Reuse validated repeated measures analysis",
            original_estimate=f"target controlled dose coefficient {guide['coefficient']:.4f}",
            original_denominator=f"{int(guide['n_rows'])} paired guide contrasts from {int(guide['n_targets'])} targets",
            strengthened_estimate="No new fit; the population dose association is retained",
            strengthened_denominator="29130 guide observations and 14565 target state pairs from 5756 targets",
            confidence_interval=f"{guide['ci_low']:.4f} to {guide['ci_high']:.4f}",
            corrected_significance=f"q {guide['q_value']:.4f}",
            effect_size_interpretation="Stronger guide perturbation is associated with a larger distal count response within target, but absolute held out prediction remains poor.",
            sensitivity_and_influence="Matched guide permutations, target bootstrap, measurement error definitions and target influence checks are already complete.",
            evidence_gained="No additional analysis was justified because every eligible target state has only two guides.",
            remaining_limitation="Target specific thresholds, saturation, donor held out guide effects and signed guide vectors are underpowered or unavailable.",
            final_status="supported",
            materially_improves_manuscript=False,
        ),
        row(
            result_family="Cross state and cross system conservation",
            extension_action="Reuse internal decomposition and stop external extension",
            original_estimate=f"mean conserved core fraction {core['estimate']:.4f}",
            original_denominator=f"{int(core['targets'])} nonzero targets within the full 4399 target cohort",
            strengthened_estimate=f"guide delta R2 {core_ranges['mean_guide_concordance']:.4f}; donor {core_ranges['mean_donor_concordance']:.4f}; K562 {core_ranges['k562_concordance']:.4f}",
            strengthened_denominator="3720 guide, 948 donor and 1204 K562 evaluable targets",
            confidence_interval=f"core fraction {core['ci_low']:.4f} to {core['ci_high']:.4f}; every predictive delta interval crosses zero",
            corrected_significance="core structure supported; predictive additions unresolved after BH correction",
            effect_size_interpretation="A conserved component is measurable within CD4 states but does not improve held out guide, donor or K562 prediction.",
            sensitivity_and_influence="Target held out, leave one state, matched target and fixed component resolution checks are complete.",
            evidence_gained="No new external evidence passed the frozen data gate.",
            remaining_limitation="The independent T cell comparison has eight targets and complete external component vectors are unavailable.",
            final_status="mixed",
            materially_improves_manuscript=False,
        ),
        row(
            result_family="External perturbation transportability",
            extension_action="Stop because no untouched outcome passed the gate",
            original_estimate=f"RPE1 full model delta R2 {external['estimate']:.4f}; vector ablation R2 {external_models.loc['without_vector', 'r2']:.4f}",
            original_denominator="490 previously inspected RPE1 targets",
            strengthened_estimate="No untouched confirmation estimate",
            strengthened_denominator="0 eligible untouched targets",
            confidence_interval=f"RPE1 delta R2 {external['ci_low']:.4f} to {external['ci_high']:.4f}",
            corrected_significance="unresolved developmental comparison; confirmatory branch unavailable",
            effect_size_interpretation="Vector ablation improves the inspected RPE1 model, but the reduced specification has not been confirmed in an untouched system.",
            sensitivity_and_influence="RPE1 ablations, calibration, coverage, shift diagnostics and target bootstrap are retained without retuning.",
            evidence_gained="The availability gate prevents RPE1 from being relabelled as confirmatory evidence.",
            remaining_limitation="The frozen KOLF object could not pass storage and complete checksum opening gates.",
            final_status="unavailable",
            materially_improves_manuscript=False,
        ),
        row(
            result_family="GRN benchmarking",
            extension_action="Paired target contrasts, calibration and heterogeneity",
            original_estimate="finite horizon transfer did not lead target macro AUROC in any eligible dataset",
            original_denominator="4 eligible datasets and 2796 evaluable targets; mESC unavailable",
            strengthened_estimate=f"random effects AUROC difference {grn_auroc['random_effect']:.4f}; {simple_wins} corrected simple method wins and {transfer_wins} corrected transfer wins across 12 contrasts",
            strengthened_denominator="1203578 identical candidate edge pairs and 2796 evaluable targets",
            confidence_interval=f"AUROC difference {grn_auroc['random_effect_ci_low']:.4f} to {grn_auroc['random_effect_ci_high']:.4f}",
            corrected_significance="simple comparators lead corrected AUROC in 3 of 4 datasets; finite horizon superiority criterion failed for all metrics",
            effect_size_interpretation="Simple correlation or pseudotime comparators outperform the transfer operator on AUROC overall, with metric specific exceptions and marked dataset heterogeneity.",
            sensitivity_and_influence=f"paired target bootstrap, target held out calibration and random effects synthesis; AUROC I2 {grn_auroc['i_squared_percent']:.1f} percent.",
            evidence_gained="Converted unpaired benchmark summaries into paired target level evidence with calibration and cross dataset heterogeneity.",
            remaining_limitation="Only four datasets are estimable and heterogeneity is high; mESC remains unavailable.",
            final_status="failed",
            materially_improves_manuscript=True,
        ),
        row(
            result_family="Molecular regulatory support",
            extension_action="Reuse complete evidence tiers and stop direct occupancy extension",
            original_estimate=f"support association with transfer gain {molecular_gain['estimate']:.4f}",
            original_denominator="124798 edges from 284 regulators and 715 regulator state combinations",
            strengthened_estimate=f"guide delta R2 {molecular_ranges['guide_concordance']:.4f}; K562 delta R2 {molecular_ranges['k562_replication']:.4f}",
            strengthened_denominator="162 guide and 99 K562 target comparisons",
            confidence_interval=f"support association {molecular_gain['ci_low']:.4f} to {molecular_gain['ci_high']:.4f}; both predictive delta intervals cross zero",
            corrected_significance=f"support association q {molecular_gain['q_value']:.4f}; predictive additions unresolved",
            effect_size_interpretation="Molecular support tracks transfer gain, but does not yet improve held out guide or K562 prediction.",
            sensitivity_and_influence="Matched backgrounds, degree rewiring, leave one resource and target held out models are complete.",
            evidence_gained="The complete denominator and correlated resource accounting prevent isolated motif or database hits from becoming mechanistic claims.",
            remaining_limitation="No compatible state matched factor occupancy exists; zero edges reach the strongest evidence tier.",
            final_status="mixed",
            materially_improves_manuscript=False,
        ),
        row(
            result_family="Natural genetic concordance",
            extension_action="Reuse corrected bundle preserving audit",
            original_estimate=f"direction agreement {natural['estimate']:.4f}",
            original_denominator=f"{int(natural['pairs'])} pairs, {int(natural['snps'])} variants, {int(natural['mediators'])} mediators and {int(natural['source_chromosomes'])} source chromosomes",
            strengthened_estimate="No new fit; corrected audit result retained",
            strengthened_denominator="425 identity eligible pairs from 31312 preselection associations",
            confidence_interval=f"{natural['bootstrap_ci_low']:.4f} to {natural['bootstrap_ci_high']:.4f}",
            corrected_significance=f"fixed 16 slot q {natural['permutation_q_fixed_16']:.4f}",
            effect_size_interpretation="Observed directional agreement is above one half, but uncertainty and the matched null do not support a firm concordance claim.",
            sensitivity_and_influence="Chromosome cluster bootstrap, symmetric eligibility, matched bundle permutations and leave one chromosome checks are complete.",
            evidence_gained="No further stratification was added because sparse matching cells would reduce validity.",
            remaining_limitation="Only 425 pairs are identity eligible and the corrected interval includes one half.",
            final_status="unresolved",
            materially_improves_manuscript=False,
        ),
        row(
            result_family="Disease convergence and locus causal triangulation",
            extension_action="Reuse complete denominator and stop unavailable genome wide extension",
            original_estimate=f"transfer augmented minus size state AUROC {disease['transfer_minus_size_state_auroc']:.4f}; 0 of {int(disease['amplified_disease_tests'])} amplified tests pass FDR",
            original_denominator=f"{int(disease['diseases'])} diseases, {int(disease['clusters'])} programmes and {len(loci)} eligible loci",
            strengthened_estimate="locus gate counts 4 of 7, 4 of 7 and 3 of 7; no mediation claim",
            strengthened_denominator="3 frozen loci with all alternate genes and unavailable gates retained",
            confidence_interval="No eligible corrected enrichment or mediation interval supports convergence",
            corrected_significance="disease prediction failed; genome wide confirmation unavailable after 0 of 7 GWAS passed schema gates",
            effect_size_interpretation="Transfer features do not improve broad disease prediction and no locus passes the causal chain.",
            sensitivity_and_influence="Matched negative diseases and loci, alternate genes, MHC exclusion and seven causal gates are retained.",
            evidence_gained="Complete gate accounting prevents partial locus evidence from being interpreted as mediation.",
            remaining_limitation="Complete ancestry matched regional GWAS, LD and immune cis eQTL inputs are unavailable.",
            final_status="failed",
            materially_improves_manuscript=False,
        ),
        row(
            result_family="Specification curve robustness and evidence synthesis",
            extension_action="Deterministic synthesis without new testing",
            original_estimate=f"buffering support {scalar.loc['buffering', 'q_supported_fraction']:.3f}; transfer gain {scalar.loc['transfer_gain', 'q_supported_fraction']:.3f}; cross system conservation {validation.loc['cross_system_conservation', 'q_supported_fraction']:.3f}; natural genetics {validation.loc['natural_genetic_concordance', 'q_supported_fraction']:.3f}",
            original_denominator="17280 scalar, 180 rerouting and 1008 validation specifications",
            strengthened_estimate="one supported stability extension, one opposite direction rerouting falsification and one failed GRN superiority extension",
            strengthened_denominator="all 10 result families with source specific denominators",
            confidence_interval="Source analysis intervals retained without recycling correlated evidence",
            corrected_significance="No new omnibus significance test; each source family retains its frozen multiplicity correction",
            effect_size_interpretation="The combined evidence supports stable within system transfer ranking and buffering, while limiting excess rerouting, general GRN superiority, portability and disease claims.",
            sensitivity_and_influence="All unavailable, underpowered, negative and contradictory branches are retained and no additional thresholds or datasets were searched.",
            evidence_gained="A single auditable table aligns estimates, denominators, uncertainty, multiplicity, stopping rules and manuscript impact.",
            remaining_limitation="Independent external perturbation, complete occupancy and genome wide disease resources remain unavailable.",
            final_status="mixed",
            materially_improves_manuscript=True,
        ),
    ]

    table = pd.DataFrame.from_records(records)
    if len(table) != 10 or table["result_family"].nunique() != 10:
        raise RuntimeError("Strengthening synthesis must contain exactly 10 unique result families")
    allowed = {"supported", "mixed", "failed", "unresolved", "unavailable", "underpowered"}
    if not set(table["final_status"]).issubset(allowed):
        raise RuntimeError("Unexpected result status")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "strengthening_results.tsv", sep="\t", index=False, lineterminator="\n")
    table.to_csv(TABLE, sep="\t", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
