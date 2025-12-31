"""Run the frozen CLR/Aitchison signed-rerouting sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses/vectors/results"
CONFIG = ROOT / "config/strengthening_extensions.yaml"
TENSOR = OUT / "contextual_transfer_tensor.parquet"
PAIRS = OUT / "context_rerouting_pairs.parquet"

# The handoff named analyses/primary/results/network_gain_rows.parquet, but the
# frozen manifest and the producing vector-analysis script both identify this
# canonical path.  The resolution is retained in the audit output.
GAIN_ROWS = OUT / "network_gain_rows.parquet"
REQUESTED_GAIN_ROWS = ROOT / "analyses/primary/results/network_gain_rows.parquet"

PAIR_OUTPUT = OUT / "composition_sensitivity_pairs.parquet"
SUMMARY_OUTPUT = OUT / "composition_sensitivity_summary.csv"
NULL_OUTPUT = OUT / "composition_sensitivity_null.parquet"
AUDIT_OUTPUT = OUT / "composition_sensitivity_audit.json"

PARTS = 30
DELTA = 1.0 / 1800.0
CLOSURE_TOLERANCE = 1e-6
EXPECTED_PAIRS = 8168
EXPECTED_TARGETS = 3696
SCOPES = ["overall", "Rest__Stim8hr", "Rest__Stim48hr", "Stim8hr__Stim48hr"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment for one finite family."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("BH family contains a non-finite P value")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(values), dtype=float)
    result[order] = np.minimum(ranked, 1.0)
    return result


def quintile_within_state_pair(frame: pd.DataFrame, column: str) -> pd.Series:
    """Create deterministic, equally populated quintiles within state pair."""
    return (
        frame.groupby("state_pair", sort=True)[column]
        .transform(lambda values: pd.qcut(values.rank(method="first"), 5, labels=False))
        .astype(np.int8)
    )


def multiplicative_replacement(compositions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply fixed-delta multiplicative zero replacement to closed rows."""
    values = np.asarray(compositions, dtype=float).copy()
    if values.ndim != 2 or values.shape[1] != PARTS:
        raise RuntimeError(f"Expected an n-by-{PARTS} composition matrix")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("Input compositions contain negative or non-finite coordinates")
    closure = values.sum(axis=1)
    if np.any(closure <= 0):
        raise RuntimeError("An evaluable composition has zero total energy")
    maximum_input_error = float(np.max(np.abs(closure - 1.0)))
    if maximum_input_error > CLOSURE_TOLERANCE:
        raise RuntimeError(
            f"Input closure check failed: maximum absolute error {maximum_input_error}"
        )

    values /= closure[:, None]
    zeros = values == 0
    zero_count = zeros.sum(axis=1)
    if np.any(zero_count * DELTA >= 1.0):
        raise RuntimeError("Fixed delta is too large for an observed zero pattern")
    values *= (1.0 - zero_count * DELTA)[:, None]
    values[zeros] = DELTA
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError("Zero replacement produced a non-positive or non-finite coordinate")
    maximum_replaced_error = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
    if maximum_replaced_error > 1e-12:
        raise RuntimeError(
            f"Replaced closure check failed: maximum absolute error {maximum_replaced_error}"
        )
    return values, zero_count


def build_pair_table() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, object]]:
    pair_source = pd.read_parquet(PAIRS).reset_index(names="source_pair_row")
    gain = pd.read_parquet(
        GAIN_ROWS,
        columns=["target_contrast", "target_contrast_gene_name", "culture_condition", "response_norm"],
    ).reset_index(drop=True)
    tensor = pd.read_parquet(TENSOR)

    if int(pair_source["pair_evaluable"].sum()) != EXPECTED_PAIRS:
        raise RuntimeError("Frozen evaluable-pair count does not equal 8,168")
    pairs = pair_source.loc[pair_source["pair_evaluable"]].copy()
    pairs["state_pair"] = pairs["left_state"].astype(str) + "__" + pairs["right_state"].astype(str)
    pairs = pairs.sort_values(["state_pair", "target_contrast"], kind="stable").reset_index(drop=True)

    left_index = pairs["left_index"].to_numpy(dtype=int)
    right_index = pairs["right_index"].to_numpy(dtype=int)
    if left_index.min() < 0 or right_index.min() < 0 or max(left_index.max(), right_index.max()) >= len(gain):
        raise RuntimeError("Pair row indices fall outside the frozen gain table")
    left_gain = gain.iloc[left_index].reset_index(drop=True)
    right_gain = gain.iloc[right_index].reset_index(drop=True)
    target = pairs["target_contrast"].astype(str).to_numpy()
    if not (
        np.array_equal(target, left_gain["target_contrast"].astype(str).to_numpy())
        and np.array_equal(target, right_gain["target_contrast"].astype(str).to_numpy())
        and np.array_equal(pairs["left_state"].astype(str).to_numpy(), left_gain["culture_condition"].astype(str).to_numpy())
        and np.array_equal(pairs["right_state"].astype(str).to_numpy(), right_gain["culture_condition"].astype(str).to_numpy())
    ):
        raise RuntimeError("Pair indices do not align with target and state labels in gain rows")
    left_norm = left_gain["response_norm"].to_numpy(dtype=float)
    right_norm = right_gain["response_norm"].to_numpy(dtype=float)
    if not (np.isfinite(left_norm).all() and np.isfinite(right_norm).all()):
        raise RuntimeError("Evaluable pairs contain non-finite response norms")
    if np.any(left_norm <= 0) or np.any(right_norm <= 0):
        raise RuntimeError("Evaluable pairs contain a non-positive response norm")
    recomputed_gain = np.abs(np.log2(right_norm / left_norm))
    frozen_gain = pairs["absolute_log2_gain_ratio"].to_numpy(dtype=float)
    gain_error = float(np.max(np.abs(recomputed_gain - frozen_gain)))
    # response_norm is stored as float32 while the frozen ratio column is
    # float64, so allow the expected sub-micro-unit round-trip discrepancy.
    if gain_error > 1e-6:
        raise RuntimeError(f"Frozen gain ratio failed recomputation: maximum error {gain_error}")
    pairs["mean_response_norm"] = 0.5 * (left_norm + right_norm)

    duplicated_tensor = int(tensor.duplicated(["target_contrast", "condition", "module"]).sum())
    if duplicated_tensor:
        raise RuntimeError("Contextual transfer tensor contains duplicate target-state-module coordinates")
    if sorted(tensor["module"].unique().tolist()) != list(range(1, PARTS + 1)):
        raise RuntimeError("Contextual transfer tensor does not contain the frozen 30 modules")
    wide = tensor.pivot(
        index=["target_contrast", "condition"], columns="module", values="energy_fraction"
    ).sort_index(axis=1)
    if wide.isna().any().any():
        raise RuntimeError("Contextual transfer tensor is incomplete after pivoting")
    module_count_distribution = tensor.groupby(["target_contrast", "condition"])["module"].nunique()
    if not module_count_distribution.eq(PARTS).all():
        raise RuntimeError("At least one target-state composition lacks 30 module parts")

    left_keys = pd.MultiIndex.from_arrays(
        [pairs["target_contrast"].astype(str), pairs["left_state"].astype(str)],
        names=wide.index.names,
    )
    right_keys = pd.MultiIndex.from_arrays(
        [pairs["target_contrast"].astype(str), pairs["right_state"].astype(str)],
        names=wide.index.names,
    )
    left_position = wide.index.get_indexer(left_keys)
    right_position = wide.index.get_indexer(right_keys)
    if np.any(left_position < 0) or np.any(right_position < 0):
        raise RuntimeError("At least one evaluable pair is missing a tensor composition")

    unique_position, compact_inverse = np.unique(
        np.concatenate([left_position, right_position]), return_inverse=True
    )
    compositions = wide.iloc[unique_position].to_numpy(dtype=float, copy=True)
    input_closure = compositions.sum(axis=1)
    replaced, zero_count = multiplicative_replacement(compositions)
    clr = np.log(replaced)
    clr -= clr.mean(axis=1, keepdims=True)
    if not np.isfinite(clr).all():
        raise RuntimeError("CLR transformation produced a non-finite coordinate")
    maximum_clr_mean = float(np.max(np.abs(clr.mean(axis=1))))
    if maximum_clr_mean > 1e-12:
        raise RuntimeError(f"CLR centering check failed: maximum row mean {maximum_clr_mean}")

    n_pairs = len(pairs)
    left_compact = compact_inverse[:n_pairs]
    right_compact = compact_inverse[n_pairs:]
    left_clr = clr[left_compact]
    right_clr = clr[right_compact]
    difference = left_clr - right_clr
    raw_distance = np.sqrt(np.sum(np.square(difference), axis=1))
    normalized_distance = raw_distance / np.sqrt(PARTS)
    if not np.isfinite(normalized_distance).all() or np.any(normalized_distance < 0):
        raise RuntimeError("Observed normalized Aitchison distance check failed")

    pairs["target_gene"] = left_gain["target_contrast_gene_name"].astype(str).to_numpy()
    pairs["gain_quintile"] = quintile_within_state_pair(pairs, "absolute_log2_gain_ratio")
    pairs["response_norm_quintile"] = quintile_within_state_pair(pairs, "mean_response_norm")
    pairs["match_stratum"] = (
        pairs["state_pair"]
        + "|g"
        + pairs["gain_quintile"].astype(str)
        + "|r"
        + pairs["response_norm_quintile"].astype(str)
    )
    pairs["left_zero_parts"] = zero_count[left_compact]
    pairs["right_zero_parts"] = zero_count[right_compact]
    pairs["left_input_closure"] = input_closure[left_compact]
    pairs["right_input_closure"] = input_closure[right_compact]
    pairs["left_replaced_closure"] = replaced[left_compact].sum(axis=1)
    pairs["right_replaced_closure"] = replaced[right_compact].sum(axis=1)
    pairs["aitchison_distance"] = raw_distance
    pairs["normalized_aitchison_distance"] = normalized_distance

    if len(pairs) != EXPECTED_PAIRS or pairs["target_contrast"].nunique() != EXPECTED_TARGETS:
        raise RuntimeError("Frozen pair or target denominator changed")
    if pairs.duplicated(["target_contrast", "state_pair"]).any():
        raise RuntimeError("An evaluable target-state pair is duplicated")
    state_pair_counts = pairs.groupby("state_pair").size().sort_index()
    if len(state_pair_counts) != 3 or (state_pair_counts < 250).any():
        raise RuntimeError("A state pair failed the frozen 250-pair minimum-power gate")
    stratum_sizes = pairs.groupby("match_stratum").size()
    if (stratum_sizes < 2).any():
        raise RuntimeError("A matching stratum cannot permute because it has fewer than two pairs")

    checks = {
        "source_pair_rows": int(len(pair_source)),
        "source_evaluable_pairs": int(pair_source["pair_evaluable"].sum()),
        "evaluated_pairs": int(len(pairs)),
        "evaluated_targets": int(pairs["target_contrast"].nunique()),
        "evaluated_compositions": int(len(unique_position)),
        "state_pair_counts": {key: int(value) for key, value in state_pair_counts.items()},
        "matching_strata": int(len(stratum_sizes)),
        "minimum_stratum_size": int(stratum_sizes.min()),
        "maximum_stratum_size": int(stratum_sizes.max()),
        "tensor_coordinate_duplicates": duplicated_tensor,
        "tensor_parts_per_target_state": PARTS,
        "maximum_input_closure_error": float(np.max(np.abs(input_closure - 1.0))),
        "maximum_replaced_closure_error": float(np.max(np.abs(replaced.sum(axis=1) - 1.0))),
        "maximum_absolute_clr_row_mean": maximum_clr_mean,
        "nonfinite_clr_coordinates": int((~np.isfinite(clr)).sum()),
        "nonfinite_observed_distances": int((~np.isfinite(normalized_distance)).sum()),
        "negative_observed_distances": int((normalized_distance < 0).sum()),
        "maximum_gain_recalculation_error": gain_error,
        "zero_replaced_coordinates": int((compositions == 0).sum()),
        "minimum_zero_parts": int(zero_count.min()),
        "maximum_zero_parts": int(zero_count.max()),
    }
    return pairs, left_clr, right_clr, checks


def run_null(
    pairs: pd.DataFrame,
    left_clr: np.ndarray,
    right_clr: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strata = [np.asarray(index, dtype=int) for index in pairs.groupby("match_stratum", sort=True).indices.values()]
    scope_indices = {
        "overall": np.arange(len(pairs), dtype=int),
        **{
            state_pair: np.flatnonzero(pairs["state_pair"].eq(state_pair).to_numpy())
            for state_pair in SCOPES[1:]
        },
    }
    records: list[dict[str, object]] = []
    for replicate in range(replicates):
        permuted_index = np.arange(len(pairs), dtype=int)
        for index in strata:
            permuted_index[index] = rng.permutation(index)
        distance = np.sqrt(np.mean(np.square(left_clr - right_clr[permuted_index]), axis=1))
        if not np.isfinite(distance).all():
            raise RuntimeError(f"Permutation {replicate} produced a non-finite distance")
        fixed = permuted_index == np.arange(len(pairs))
        for scope in SCOPES:
            index = scope_indices[scope]
            records.append(
                {
                    "scope": scope,
                    "replicate": replicate,
                    "pairs": int(len(index)),
                    "mean_normalized_aitchison_distance": float(np.mean(distance[index])),
                    "median_normalized_aitchison_distance": float(np.median(distance[index])),
                    "fixed_points": int(fixed[index].sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def cluster_bootstrap(
    frame: pd.DataFrame,
    null_median: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample target clusters and subtract the fixed permutation median."""
    target = (
        frame.groupby("target_contrast", sort=True)["normalized_aitchison_distance"]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )
    sums = target["sum"].to_numpy(dtype=float)
    counts = target["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.integers(0, len(target), len(target))
        draws[replicate] = sums[selected].sum() / counts[selected].sum() - null_median
    if not np.isfinite(draws).all():
        raise RuntimeError("Target-cluster bootstrap produced a non-finite estimate")
    return draws


def summarize(
    pairs: pd.DataFrame,
    null: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in SCOPES:
        frame = pairs if scope == "overall" else pairs.loc[pairs["state_pair"].eq(scope)]
        null_scope = null.loc[null["scope"].eq(scope), "mean_normalized_aitchison_distance"]
        observed_mean = float(frame["normalized_aitchison_distance"].mean())
        observed_median = float(frame["normalized_aitchison_distance"].median())
        null_median = float(null_scope.median())
        effect = observed_mean - null_median
        draws = cluster_bootstrap(
            frame,
            null_median,
            bootstrap_replicates,
            seed,
        )
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        p_upper = float((1 + np.sum(null_scope.to_numpy(dtype=float) >= observed_mean)) / (len(null_scope) + 1))
        p_lower = float((1 + np.sum(null_scope.to_numpy(dtype=float) <= observed_mean)) / (len(null_scope) + 1))
        required_pairs = 1000 if scope == "overall" else 250
        rows.append(
            {
                "scope": scope,
                "pairs": int(len(frame)),
                "targets": int(frame["target_contrast"].nunique()),
                "observed_mean_normalized_aitchison_distance": observed_mean,
                "observed_median_normalized_aitchison_distance": observed_median,
                "permutation_null_median_of_means": null_median,
                "permutation_null_mean_of_means": float(null_scope.mean()),
                "permutation_null_ci_low": float(null_scope.quantile(0.025)),
                "permutation_null_ci_high": float(null_scope.quantile(0.975)),
                "observed_minus_null_median": effect,
                "cluster_bootstrap_ci_low": float(ci_low),
                "cluster_bootstrap_ci_high": float(ci_high),
                "permutation_p_upper": p_upper,
                "permutation_p_lower": p_lower,
                "minimum_required_pairs": required_pairs,
                "passes_minimum_power": bool(len(frame) >= required_pairs),
                "bootstrap_replicates": bootstrap_replicates,
                "permutation_replicates": int(len(null_scope)),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    summary["q_value_upper_bh"] = bh(summary["permutation_p_upper"].to_numpy(dtype=float))
    supported = (
        summary["passes_minimum_power"]
        & summary["cluster_bootstrap_ci_low"].gt(0)
        & summary["q_value_upper_bh"].lt(0.05)
    )
    failed = summary["passes_minimum_power"] & summary["cluster_bootstrap_ci_high"].lt(0)
    summary["status"] = np.where(
        ~summary["passes_minimum_power"],
        "underpowered",
        np.where(supported, "supported", np.where(failed, "failed", "mixed")),
    )
    summary["effect_direction"] = np.where(
        summary["observed_minus_null_median"].gt(0),
        "observed_greater_than_matched_null",
        "observed_less_than_matched_null",
    )
    return summary


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    specification = config["families"]["signed_rerouting_context"]
    seed = int(config["seed"])
    bootstrap_replicates = int(config["bootstrap_replicates"])
    permutation_replicates = int(config["permutation_replicates"])
    if seed != 20260912 or bootstrap_replicates != 1000 or permutation_replicates != 1000:
        raise RuntimeError("Frozen seed or replicate counts changed")
    if "8168" not in str(specification["eligible_denominator"]).replace(",", ""):
        raise RuntimeError("Frozen signed-rerouting denominator changed")
    if not (TENSOR.exists() and PAIRS.exists() and GAIN_ROWS.exists()):
        raise FileNotFoundError("A canonical frozen vector input is missing")

    pairs, left_clr, right_clr, checks = build_pair_table()
    null = run_null(pairs, left_clr, right_clr, permutation_replicates, seed)
    summary = summarize(pairs, null, bootstrap_replicates, seed)

    overall = summary.loc[summary["scope"].eq("overall")].iloc[0]
    decision = str(overall["status"])
    pairs.to_parquet(PAIR_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    null.to_parquet(NULL_OUTPUT, index=False)

    audit = {
        "analysis": "frozen_signed_rerouting_composition_sensitivity",
        "status": decision,
        "hypothesis": specification["hypothesis"],
        "decision_criterion": specification["decision_criterion"],
        "decision_criterion_met": bool(decision == "supported"),
        "interpretation": (
            "Same-target cross-state programme compositions were closer than matched cross-target "
            "permutations under the frozen CLR geometry."
            if float(overall["observed_minus_null_median"]) < 0
            else "Same-target cross-state programme compositions were farther apart than the matched null."
        ),
        "scalar_context_result": {
            "status": "failed",
            "rule": specification["scalar_context_rule"],
            "preserved_without_reinterpretation": True,
        },
        "configuration": {
            "parts": PARTS,
            "zero_replacement": "multiplicative",
            "delta": DELTA,
            "distance": "Euclidean CLR distance divided by sqrt(30)",
            "summary_statistic": "mean normalized Aitchison distance",
            "matching": "right-state target permutation within state pair and within-state-pair quintiles of absolute gain and mean response norm",
            "quintile_tie_rule": "rank(method='first') after deterministic state-pair/target sorting",
            "permutation_fixed_points": "allowed by the frozen unrestricted within-stratum permutation",
            "bootstrap": "target-cluster percentile bootstrap of observed mean minus fixed permutation-null median",
            "multiplicity": "Benjamini-Hochberg over overall plus the three state-pair directional upper-tail tests",
            "seed": seed,
            "permutation_replicates": permutation_replicates,
            "bootstrap_replicates": bootstrap_replicates,
            "confidence_level": float(config["confidence_level"]),
        },
        "input_path_resolution": {
            "handoff_path": str(REQUESTED_GAIN_ROWS.relative_to(ROOT)).replace("\\", "/"),
            "handoff_path_exists": REQUESTED_GAIN_ROWS.exists(),
            "canonical_path_used": str(GAIN_ROWS.relative_to(ROOT)).replace("\\", "/"),
            "reason": "The vector-analysis producer and frozen strengthening manifest identify the canonical vector-results path.",
        },
        "inputs": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [TENSOR, PAIRS, GAIN_ROWS]
        ],
        "checks": checks,
        "results": json.loads(summary.to_json(orient="records")),
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path, rows in [
                (PAIR_OUTPUT, len(pairs)),
                (SUMMARY_OUTPUT, len(summary)),
                (NULL_OUTPUT, len(null)),
            ]
        ],
        "limitations": [
            "The target-cluster interval treats the permutation-null median as fixed and therefore does not include Monte Carlo uncertainty in that median.",
            "Inference is conditional on the frozen 30-component reference decomposition and the prespecified 1/1800 zero replacement.",
            "The available tensor is target-state level, so guide- and donor-specific compositional reproducibility cannot be estimated here.",
            "Matched permutations preserve coarse quintiles rather than exact continuous gain and response-norm values.",
        ],
    }
    AUDIT_OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision, "outputs": audit["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
