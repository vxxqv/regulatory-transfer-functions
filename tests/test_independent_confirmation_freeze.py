import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/independent_confirmation.yaml"
FREEZE = ROOT / "analyses/independent_confirmation"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".tsv", ".txt", ".yaml", ".yml"}


def config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def canonical_hash(path):
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        data = text.encode("utf-8")
        return hashlib.sha256(data).hexdigest(), len(data), "text_utf8_lf_sha256"
    return hashlib.sha256(data).hexdigest(), len(data), "binary_exact_sha256"


def test_metadata_only_registry_and_two_stage_hash_gate():
    cfg = config()
    assert cfg["status"] == "frozen_before_independent_outcome_inspection"
    assert cfg["scope"]["outcomes_must_remain_unopened_until_all_gates_pass"] is True
    assert cfg["scope"]["selection_must_be_outcome_blind"] is True
    assert cfg["scope"]["metadata_only_screening_allowed"] is True
    forbidden = set(cfg["scope"]["metadata_screening_prohibitions"])
    assert {"expression_matrix_values", "response_vector_values", "effect_sizes", "p_values", "q_values"} <= forbidden
    stages = cfg["dataset_selection"]["two_stage_hash_process"]
    assert stages["stage_1_before_selection"]["name"] == "provisional_metadata_hash"
    assert stages["stage_1_before_selection"]["outcome_content_forbidden"] is True
    assert stages["stage_2_before_outcome_opening"]["require_full_file_sha256"] is True
    assert stages["stage_2_before_outcome_opening"]["require_provider_checksum_when_available"] is True
    assert stages["stage_2_before_outcome_opening"]["opening_allowed_only_after_checksum_and_role_lock"] is True
    immutable = set(cfg["dataset_selection"]["immutable_after_selection_fields"])
    assert {"dataset_accession", "dataset_role", "mapping_sha256", "outcome_unopened", "provisional_metadata_sha256"} <= immutable


def test_candidate_template_has_schema_but_no_outcome_data():
    path = FREEZE / "candidate_manifest_template.tsv"
    rows = read_tsv(path)
    assert rows == []
    header = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    required = {
        "candidate_id",
        "block",
        "dataset_accession",
        "provisional_metadata_sha256",
        "mapping_sha256",
        "dataset_role",
        "outcome_unopened",
        "final_full_file_sha256",
        "provider_checksum",
        "final_checksum_verified",
        "outcome_opening_gate_status",
    }
    assert required <= set(header)
    prohibited = {
        "expression_value",
        "response_value",
        "effect_size",
        "p_value",
        "q_value",
        "prediction",
        "phenotype",
        "outcome_value",
    }
    assert prohibited.isdisjoint(header)


def test_dataset_roles_exclude_all_inspected_systems():
    rows = read_tsv(FREEZE / "dataset_roles.tsv")
    keyed = {row["dataset_key"]: row for row in rows}
    assert {"GSE314342_CD4", "GSE314342_Th1_Th2_eight_target", "Replogle_K562", "Replogle_RPE1", "ALL_PROJECT_OUTCOMES_INSPECTED_BEFORE_FREEZE"} <= set(keyed)
    for row in rows:
        if row["outcomes_inspected_before_freeze"] == "true":
            assert row["outcome_unopened"] == "false"
            assert row["eligible_for_untouched_portability"] == "false"
            assert row["eligible_for_definitive_primary_t_cell_replication"] == "false"
            assert row["immutable_role"] in {"development_source", "previously_inspected_excluded"}
    assert keyed["GSE314342_Th1_Th2_eight_target"]["eligible_for_definitive_primary_t_cell_replication"] == "false"
    assert keyed["METADATA_ONLY_SCREENED_POOL"]["outcome_unopened"] == "true"
    cfg = config()
    exclusions = set(cfg["global_exclusions"]["portability_confirmation_forbidden_systems"])
    assert exclusions == {"CD4", "K562", "RPE1", "Th1", "Th2"}
    assert cfg["global_exclusions"]["previously_inspected_outcomes_excluded"] is True


def test_untouched_portability_eligibility_models_and_decision_gate():
    block = config()["untouched_portability_confirmation"]
    gate = block["eligibility"]
    assert gate["species"] == "Homo_sapiens"
    assert gate["require_genetic_perturbation"] is True
    assert gate["require_transcriptomic_readout"] is True
    assert gate["require_processed_counts_or_signed_response_vectors"] is True
    assert gate["require_non_targeting_controls"] is True
    assert gate["require_unambiguous_target_mapping"] is True
    assert gate["minimum_overlapping_cd4_targets"] == 100
    assert gate["minimum_response_genes"] == 500
    assert gate["minimum_cells_per_target"] == 50
    assert gate["alternative_to_cell_minimum"] == "author_validated_pseudobulk_uncertainty"
    assert gate["outcome_blind_selection"] is True
    reuse = block["frozen_cd4_reuse"]
    assert reuse["target_folds"]["folds"] == 10
    assert "sha256" in reuse["target_folds"]["rule"]
    assert reuse["target_folds"]["regenerate"] is False
    assert reuse["identical_splits_for_every_model_and_ablation"] is True
    assert reuse["feature_selection"] == "source_cd4_training_data_only"
    assert "covariate_shift_fit" in reuse["heldout_outcomes_forbidden_for"]
    assert set(block["models"]["primary"]) == {"frozen_cd4_transfer", "without_vector"}
    assert set(block["models"]["competitive_baselines"]) == {"global_mean", "simple_covariates", "ridge", "sparse_nonlinear", "standard_network_propagation", "graph_readout"}
    assert "vector" not in block["models"]["without_vector_feature_blocks"]
    assert block["covariate_shift"]["heldout_outcomes_used"] is False
    assert block["inference"]["paired_comparison_unit"] == "target"
    assert block["inference"]["interval"] == "paired_target_bootstrap"
    assert block["inference"]["calibration_slope_acceptable"] == [0.8, 1.2]
    assert block["inference"]["coverage_levels"] == [0.5, 0.8, 0.95]
    assert block["inference"]["maximum_absolute_coverage_error"] == 0.05
    assert len(block["inference"]["fixed_target_strata"]) == 5
    decision = block["primary_pass"]
    assert decision["model"] == "without_vector"
    assert decision["require_paired_95_ci_excludes_zero_above"] is True
    assert decision["require_within_family_q_below"] == 0.05
    assert decision["require_acceptable_calibration"] is True
    assert decision["require_acceptable_coverage"] is True
    assert decision["require_fixed_strata_reproducibility"] is True


def test_primary_t_cell_replication_power_and_independence_gates():
    block = config()["independent_primary_t_cell_state_replication"]
    gate = block["per_dataset_eligibility"]
    assert block["dataset_count_minimum"] == 2
    assert block["independence_definition"]["distinct_study_accession"] is True
    assert block["independence_definition"]["nonoverlapping_participant_or_donor_cohort"] is True
    assert gate["require_primary_t_cells"] is True
    assert gate["minimum_matched_activation_states"] == 2
    assert gate["require_non_targeting_controls"] is True
    assert gate["minimum_frozen_mapped_targets"] == 50
    assert gate["minimum_cells_per_target_state"] == 30
    assert gate["alternative_to_cell_minimum"] == "validated_pseudobulk_uncertainty"
    assert gate["minimum_common_genes"] == 500
    assert gate["mapping_frozen_before_outcomes"] is True
    assert gate["minimum_repeated_units"] == 3
    assert gate["repeated_units_must_span_each_state"] is True
    assert "GSE314342_Th1_Th2_eight_target_validation" in block["excluded_from_definitive_replication"]


def test_guide_by_donor_gate_model_and_null_families():
    block = config()["guide_by_donor_reconstruction"]
    gate = block["outcome_opening_gate"]
    assert gate["require_raw_or_processed_guide_assignments"] is True
    assert gate["require_donor_labels"] is True
    assert gate["require_non_targeting_controls"] is True
    assert gate["minimum_distinct_supported_guide_efficacies"] == 3
    assert gate["minimum_donors_for_donor_heldout_claim"] == 3
    assert gate["minimum_cells_per_guide_donor"] == 30
    assert gate["minimum_response_genes"] == 500
    assert set(block["models"]) == {"null", "linear", "restricted_cubic_spline", "isotonic", "breakpoint", "hill"}
    assert set(block["validation"]) == {"donor_held_out", "guide_held_out"}
    assert {"measurement_error", "calibration", "guide_bootstrap", "donor_bootstrap", "matched_guide_permutations"} <= set(block["required_sensitivities"])
    assert set(block["controls"]) == {"non_targeting_controls", "negative_control_genes"}
    assert block["claim_prohibitions"]["threshold_or_saturation_with_fewer_than_three_supported_doses"] is True


def test_molecular_evidence_factor_background_tier_and_null_freeze():
    block = config()["state_compatible_molecular_evidence"]
    assert all(block["eligibility"].values())
    factor = block["factor_freeze"]
    assert factor["factor_lists_frozen_before_occupancy_inspection"] is True
    assert factor["privileged_pair_forbidden"] == "GATA3_EGR1"
    assert factor["no_named_pair_privileged"] is True
    resources = set(block["eligible_resources"])
    assert {"state_matched_ChIP_seq", "state_matched_CUT_and_RUN", "state_matched_ATAC_seq", "state_matched_H3K27ac", "promoter_capture_HiC", "validated_enhancer_to_gene_links", "JASPAR", "HOCOMOCO", "DoRothEA", "decoupleR", "RegNetwork", "TRRUST", "TFTG", "CollecTRI", "OmniPath", "independent_genetic_perturbations"} <= resources
    assert block["correlated_curated_databases_count_as_one_evidence_type"] is True
    assert set(block["matched_background_covariates"]) == {"expression", "accessibility", "gc_content", "interval_length", "peak_count", "regulator_target_distance", "response_degree", "network_degree"}
    assert set(block["evidence_tiers"]) == {"tier_1", "tier_2", "tier_3", "tier_4", "unavailable"}
    assert block["evidence_tiers"]["tier_3"]["directional_conflict_must_be_reported"] is True
    assert set(block["hypotheses"]) == {"direct_edge", "cascade_timing"}
    assert block["nulls"]["matched_permutations"] == 1000
    assert block["nulls"]["degree_preserving_rewirings"] == 1000


def test_genome_wide_genetics_and_mediation_gates():
    block = config()["genome_wide_disease_genetics"]
    assert all(block["opening_gate"].values())
    assert block["credible_sets"]["posterior_mass"] == 0.95
    assert block["credible_sets"]["require_complete_locus_coverage"] is True
    assert block["colocalization"]["priors"] == {"p1": 0.0001, "p2": 0.0001, "p12": 0.00001}
    assert block["colocalization"]["minimum_pp4"] == 0.8
    assert block["mhc"]["primary_analysis"] == "excluded"
    assert block["mhc"]["sensitivity_analysis"] == "included"
    assert len(block["competitive_gene_set_covariates"]) >= 7
    assert len(block["partitioned_heritability_covariates"]) >= 6
    assert set(block["negative_controls"]) == {"matched_negative_diseases", "matched_negative_loci", "matched_negative_genes", "matched_negative_programs", "matched_negative_genomic_regions"}
    instruments = block["instruments"]
    assert instruments["minimum_f_statistic"] == 10
    assert instruments["require_steiger_directionality"] is True
    assert instruments["require_no_material_cochran_q_heterogeneity"] is True
    assert instruments["require_no_horizontal_pleiotropy"] is True
    assert block["mediation"]["allowed_only_if_all_colocalization_instrument_directionality_heterogeneity_and_pleiotropy_assumptions_pass"] is True
    assert block["mediation"]["claim_if_any_assumption_fails"] is False


def test_integrated_evidence_and_novelty_result_gate():
    cfg = config()
    block = cfg["integrated_evidence_portability"]
    assert set(block["outcomes"]) == {"guide_concordance", "cross_state_core_membership", "untouched_external_portability", "information_retention", "disease_relevance"}
    assert block["comparator"] == "covariate_only"
    assert block["validation"]["target_held_out"] is True
    assert block["validation"]["calibration_required"] is True
    assert block["nulls"]["matched_permutations"] == 1000
    assert block["nulls"]["degree_preserving_rewirings"] == 1000
    assert block["leave_one_resource_analyses"].startswith("required")
    gate = cfg["novelty_and_result_gate"]
    assert set(gate["part_1_novelty"]["require_one"]) == {"independent_evidence_source", "materially_distinct_prespecified_test"}
    assert gate["part_2_result_capability"]["must_be_capable_of_changing_or_narrowing_central_conclusion"] is True
    assert set(gate["part_2_result_capability"]["required_outputs"]) == {"effect_size", "uncertainty", "denominator", "calibration_or_falsification_context", "decision_rule"}
    assert set(gate["stop_without_substitution_if"]) == {"redundant_method", "outcome_selected_dataset", "weak_nominal_only_result", "branch_cannot_change_interpretation"}
    assert gate["favorable_p_value_alone_is_a_good_result"] is False


def test_hypotheses_are_unique_directional_and_decidable():
    rows = read_tsv(FREEZE / "hypotheses.tsv")
    identifiers = [row["hypothesis_id"] for row in rows]
    assert identifiers == [f"H{i}" for i in range(18, 30)]
    assert len(identifiers) == len(set(identifiers))
    assert {row["endpoint_tier"] for row in rows} == {"primary", "secondary"}
    for row in rows:
        assert row["directional_expectation"]
        assert row["endpoint"]
        assert row["pass_rule"]
        assert row["fail_rule"]
        assert row["unavailable_rule"]
        assert row["multiplicity_family"]
        assert row["status_at_freeze"] == "not_tested"


def test_fixed_seeds_and_resample_refinement_rule():
    inference = config()["inference"]
    seeds = inference["seeds"]
    assert len(seeds) == 9
    assert all(isinstance(value, int) for value in seeds.values())
    assert len(set(seeds.values())) == len(seeds)
    resamples = inference["resamples"]
    for name in ("default", "target_bootstrap", "guide_bootstrap", "donor_bootstrap", "clustered_bootstrap", "matched_permutations", "degree_preserving_rewirings"):
        assert resamples[name] == 1000
    refinement = resamples["refinement"]
    assert refinement["allowed_only_if_absolute_q_distance_from_0_05_at_most"] == 0.005
    assert refinement["same_seed_stream_required"] is True
    assert refinement["no_other_adaptive_increase_allowed"] is True


def test_existing_target_fold_artifact_uses_ten_sha256_folds():
    rows = read_tsv(ROOT / "analyses/external_benchmark/target_folds.tsv")
    assert rows
    assert {int(row["fold"]) for row in rows} == set(range(1, 11))
    assert len({row["target_gene"] for row in rows}) == len(rows)
    for row in rows:
        expected = int(hashlib.sha256(row["target_gene"].encode("utf-8")).hexdigest()[:16], 16) % 10 + 1
        assert int(row["fold"]) == expected


def test_manifest_hashes_every_frozen_and_reused_artifact():
    manifest_path = FREEZE / "freeze_manifest.json"
    raw_manifest = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(raw_manifest)
    assert manifest["status"] == "frozen_before_independent_outcome_inspection"
    assert manifest["new_external_outcomes_opened"] is False
    assert manifest["candidate_registry_has_outcome_rows"] is False
    assert str(ROOT) not in raw_manifest
    assert "C:\\" not in raw_manifest
    frozen_expected = {
        "config/independent_confirmation.yaml",
        "analyses/independent_confirmation/README.md",
        "analyses/independent_confirmation/freeze.py",
        "analyses/independent_confirmation/hypotheses.tsv",
        "analyses/independent_confirmation/dataset_roles.tsv",
        "analyses/independent_confirmation/candidate_manifest_template.tsv",
        "tests/test_independent_confirmation_freeze.py",
    }
    frozen = {record["path"] for record in manifest["frozen_files"]}
    assert frozen == frozen_expected
    reused = {record["path"] for record in manifest["reused_cd4_artifacts"]}
    assert {
        "config/analysis.yaml",
        "config/external_benchmark.yaml",
        "analyses/external_benchmark/target_folds.tsv",
        "analyses/primary/results/transfer_phenotypes.parquet",
        "analyses/vectors/results/contextual_transfer_tensor.parquet",
        "analyses/vectors/results/reference_module_loadings.parquet",
        "analyses/external_benchmark/run_benchmark.py",
    } <= reused
    assert manifest["target_folds"]["folds"] == 10
    assert "sha256" in manifest["target_folds"]["rule"]
    for record in manifest["frozen_files"] + manifest["reused_cd4_artifacts"]:
        relative = Path(record["path"])
        assert not relative.is_absolute()
        assert ":" not in record["path"]
        path = ROOT / relative
        digest, size, mode = canonical_hash(path)
        assert record["hash_mode"] == mode
        assert record["sha256"] == digest
        size_key = "exact_bytes" if mode == "binary_exact_sha256" else "canonical_lf_bytes"
        assert record[size_key] == size


def test_text_hashes_accept_raw_lf_and_crlf_equivalence():
    raw = b"alpha\rbravo\r\ncharlie\n"
    variants = [raw, raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")]
    variants.append(variants[1].replace(b"\n", b"\r\n"))
    hashes = set()
    for value in variants:
        normalized = value.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        hashes.add(hashlib.sha256(normalized).hexdigest())
    assert len(hashes) == 1
