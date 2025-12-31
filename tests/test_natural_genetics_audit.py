from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from analyses.natural_genetics.run_eqtlgen_audit import (
    CONDITIONS,
    FAMILY_SIZE,
    build_association_universe,
    build_target_strata,
    calculate_statistic,
    chromosome_cluster_bootstrap,
    draw_bundle_mapping,
    enumerate_bijections,
    fixed_family_bh,
    is_palindromic,
    leave_one_chromosome_out,
    mapped_effects,
    mapping_is_bundle_preserving,
    sha256_file,
    slots,
    verify_amendment,
)
from src.models.natural_genetics import allele_orientation


ROOT = Path(__file__).resolve().parents[1]


def test_completed_output_marks_partial_null_series_unavailable() -> None:
    result = pd.read_csv(
        ROOT / "analyses/natural_genetics/results_audit_corrected/hypothesis_tests.csv"
    )
    partial = result["null_finite_replicates"] < 1000
    assert partial.sum() == 3
    assert result.loc[partial, "available"].eq(False).all()
    assert result.loc[partial, "permutation_p_upper"].isna().all()
    assert result.loc[partial, "permutation_q_fixed_16"].isna().all()
    assert result.loc[partial, "unavailable_reason"].eq(
        "one_or_more_scheduled_null_replicates_undefined"
    ).all()


def toy_rows(targets: int = 6) -> pd.DataFrame:
    records = []
    for target_index in range(targets):
        for state_index, state in enumerate(CONDITIONS):
            records.append(
                {
                    "target_contrast": f"T{target_index}",
                    "culture_condition": state,
                    "target_baseMean": float(target_index + state_index + 1),
                    "n_cells_target": float(100 + target_index * 10 + state_index),
                    "response_row": len(records),
                }
            )
    return pd.DataFrame(records)


def toy_universe() -> pd.DataFrame:
    records = []
    for index, (chromosome, state, bonferroni) in enumerate(
        [(1, "Rest", 0.01), (1, "Stim8hr", 0.10), (2, "Rest", 0.01), (2, "Stim48hr", 0.10)]
    ):
        records.append(
            {
                "SNPChr_cis": chromosome,
                "condition": state,
                "BonferroniP_trans": bonferroni,
                "Zscore_cis": 1.0,
                "observed_trans_z": 1.0 if index != 1 else -1.0,
                "SNP": f"rs{index}",
                "Gene_cis": f"T{index}",
                "Gene_trans": f"G{index}",
            }
        )
    return pd.DataFrame(records)


def observed_summary(universe: pd.DataFrame, effects: np.ndarray, eligible: np.ndarray) -> pd.DataFrame:
    records = []
    for subset, scope, metric in slots():
        estimate, _ = calculate_statistic(universe, effects, eligible, subset, scope, metric)
        records.append({"subset": subset, "scope": scope, "metric": metric, "estimate": estimate})
    return pd.DataFrame(records)


def test_palindromic_variants_are_unresolved_and_excluded() -> None:
    first = pd.Series(["A", "T", "C", "G", "A"])
    second = pd.Series(["T", "A", "G", "C", "G"])
    assert is_palindromic(first, second).tolist() == [True, True, True, True, False]


def test_all_qc_median_bins_use_inclusive_low_boundary() -> None:
    rows = toy_rows(5)
    rest = rows["culture_condition"] == "Rest"
    rows.loc[rest, "target_baseMean"] = [1, 2, 3, 4, 5]
    degree = np.arange(len(rows), dtype=int)
    targets, _, thresholds, _ = build_target_strata(rows, degree, minimum_targets=5)
    assert thresholds.query("condition == 'Rest' and covariate == 'target_baseMean'")["median"].iat[0] == 3
    assert targets.set_index("target").loc["T2", "Rest_target_baseMean_bin"] == "low"
    assert targets.set_index("target").loc["T3", "Rest_target_baseMean_bin"] == "high"


def test_exact_state_availability_is_part_of_stratum() -> None:
    rows = toy_rows(2)
    rows = rows.loc[~((rows["target_contrast"] == "T1") & (rows["culture_condition"] == "Stim48hr"))]
    targets, _, _, _ = build_target_strata(rows.reset_index(drop=True), np.arange(len(rows)))
    indexed = targets.set_index("target")
    assert indexed.loc["T0", "state_availability"] == "Rest,Stim8hr,Stim48hr"
    assert indexed.loc["T1", "state_availability"] == "Rest,Stim8hr"
    assert indexed.loc["T0", "stratum_key"] != indexed.loc["T1", "stratum_key"]


def test_small_strata_are_fixed_and_large_strata_are_bijections() -> None:
    small = pd.DataFrame({"target": ["a", "b", "c", "d"], "stratum_key": ["s"] * 4})
    mapping = draw_bundle_mapping(small, np.random.default_rng(4), minimum_targets=5)
    assert mapping == {target: target for target in ["a", "b", "c", "d"]}
    large = pd.DataFrame({"target": list("abcdef"), "stratum_key": ["s"] * 6})
    mapping = draw_bundle_mapping(large, np.random.default_rng(4), minimum_targets=5)
    assert set(mapping) == set(mapping.values()) == set("abcdef")


def test_bundle_mapping_preserves_states_and_one_mediator_mapping() -> None:
    rows = toy_rows(6)
    targets, states, _, _ = build_target_strata(rows, np.ones(len(rows)), minimum_targets=2)
    mapping = draw_bundle_mapping(targets, np.random.default_rng(8), minimum_targets=2)
    assert mapping_is_bundle_preserving(mapping, targets)
    universe = pd.DataFrame(
        {
            "Gene_cis": ["T0"] * 3,
            "condition": list(CONDITIONS),
            "gene_column": [0, 0, 0],
            "Zscore_cis": [1.0] * 3,
            "observed_trans_z": [1.0] * 3,
        }
    )
    matrix = sparse.csr_matrix(np.arange(len(rows), dtype=float).reshape(-1, 1) + 1)
    _, _, response_rows = mapped_effects(universe, mapping, states, matrix)
    mapped_target = mapping["T0"]
    expected = states.loc[states["target"] == mapped_target].set_index("condition").loc[list(CONDITIONS), "response_row"]
    assert response_rows.tolist() == expected.astype(int).tolist()


def test_seeded_mapping_is_invariant_to_target_row_order() -> None:
    rows = toy_rows(8)
    targets, _, _, _ = build_target_strata(rows, np.arange(len(rows)), minimum_targets=2)
    first = draw_bundle_mapping(targets, np.random.default_rng(20260912), minimum_targets=2)
    shuffled = targets.sample(frac=1, random_state=73).reset_index(drop=True)
    second = draw_bundle_mapping(shuffled, np.random.default_rng(20260912), minimum_targets=2)
    assert first == second


def test_pre_effect_universe_retains_zero_and_eligibility_is_symmetric() -> None:
    rows = pd.DataFrame(
        {
            "target_contrast": ["T0"],
            "culture_condition": ["Rest"],
            "response_row": [0],
            "target_baseMean": [1.0],
            "n_cells_target": [100.0],
        }
    )
    genes = pd.DataFrame({"feature_id": ["G0"]})
    cis = pd.DataFrame(
        {
            "SNP": ["rs1"], "Gene": ["T0"], "SNPChr": [1], "SNPPos": [1],
            "AssessedAllele": ["A"], "OtherAllele": ["G"], "Zscore": [2.0],
        }
    )
    trans = pd.DataFrame(
        {
            "SNP": ["rs1"], "Gene": ["G0"], "SNPChr": [1], "SNPPos": [1],
            "AssessedAllele": ["A"], "OtherAllele": ["G"], "Zscore": [3.0],
            "BonferroniP": [0.01],
        }
    )
    universe, _ = build_association_universe(rows, genes, cis, trans, allele_orientation, ["Rest"])
    targets, states, _, _ = build_target_strata(rows, np.array([0]), ["Rest"], minimum_targets=1)
    mapping = {target: target for target in targets["target"]}
    effects, eligible, _ = mapped_effects(universe, mapping, states, sparse.csr_matrix([[0.0]]))
    assert len(universe) == 1 and effects[0] == 0 and not eligible[0]
    permuted_effects, permuted_eligible, _ = mapped_effects(universe, mapping, states, sparse.csr_matrix([[0.0]]))
    assert np.array_equal(eligible, permuted_eligible) and np.array_equal(effects, permuted_effects)


def test_toy_exact_enumeration_contains_every_bijection_once() -> None:
    mappings = list(enumerate_bijections(["c", "a", "b"]))
    encoded = {tuple(mapping[key] for key in ["a", "b", "c"]) for mapping in mappings}
    assert len(mappings) == 6 and len(encoded) == 6
    assert sum(mapping["a"] == "a" for mapping in mappings) == 2


def test_primary_and_bonferroni_slots_use_prespecified_subsets() -> None:
    universe = toy_universe()
    effects = np.ones(len(universe))
    eligible = np.ones(len(universe), dtype=bool)
    _, primary_n = calculate_statistic(universe, effects, eligible, "primary", "overall", "direction_agreement")
    _, bonferroni_n = calculate_statistic(universe, effects, eligible, "bonferroni", "overall", "direction_agreement")
    assert primary_n == 4 and bonferroni_n == 2
    assert len(slots()) == FAMILY_SIZE == 16


def test_fixed_family_bh_retains_unavailable_slots_in_denominator() -> None:
    p_values = [0.01, 0.02] + [np.nan] * 14
    adjusted = fixed_family_bh(p_values, family_size=16)
    assert np.allclose(adjusted[:2], [0.16, 0.16])
    assert np.isnan(adjusted[2:]).all()


def test_source_chromosome_bootstrap_resamples_whole_clusters() -> None:
    universe = pd.concat([toy_universe().iloc[[0]], toy_universe().iloc[[2, 2, 2]]], ignore_index=True)
    universe.loc[:, "SNPChr_cis"] = [1, 2, 2, 2]
    effects = np.ones(len(universe))
    eligible = np.ones(len(universe), dtype=bool)
    result = chromosome_cluster_bootstrap(
        universe, effects, eligible, "primary", "overall", "direction_agreement", 100, 20260912
    )
    assert len(result) == 100
    assert set(result["denominator"]).issubset({2, 4, 6})
    assert (result["source_chromosome_draws"] == 2).all()


def test_true_loco_removes_source_chromosome_with_frozen_mappings() -> None:
    universe = toy_universe()
    identity = np.array([1.0, 1.0, 1.0, 1.0])
    eligible = np.ones(4, dtype=bool)
    permutation_effects = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=float)
    permutation_eligible = np.ones_like(permutation_effects, dtype=bool)
    result = leave_one_chromosome_out(
        universe,
        identity,
        eligible,
        permutation_effects,
        permutation_eligible,
        observed_summary(universe, identity, eligible),
    )
    row = result.query(
        "excluded_source_chromosome == 1 and subset == 'primary' and scope == 'overall' and metric == 'direction_agreement'"
    ).iloc[0]
    assert row["denominator"] == 2
    assert row["source_chromosomes_retained"] == 1
    assert bool(row["strata_frozen"]) and bool(row["permutation_mappings_frozen"])


def test_amendment_preserves_original_hashes_and_provenance() -> None:
    path = ROOT / "analyses/natural_genetics/audit_correction_01.yaml"
    frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [relative for relative in frozen["inputs"] if not (ROOT / relative).exists()]
    amendment = frozen if missing else verify_amendment(path)
    assert amendment["status"] == "retrospective_audit_correction_frozen_before_corrected_run"
    assert amendment["design"]["seed"] == 20260912
    assert amendment["design"]["multiplicity"]["family_slots"] == 16
    for name, expected in amendment["original_outputs"]["files"].items():
        output = ROOT / "analyses/natural_genetics/results" / name
        raw = output.read_bytes()
        candidates = {sha256_file(output)}
        if output.suffix.lower() in {".csv", ".json", ".md", ".tsv", ".txt", ".yaml", ".yml"}:
            lf = raw.replace(b"\r\n", b"\n")
            candidates.update({
                hashlib.sha256(lf).hexdigest(),
                hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
            })
        assert expected in candidates
