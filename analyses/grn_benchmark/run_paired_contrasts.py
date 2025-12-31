"""Run the frozen paired GRN benchmarking extension.

This script deliberately compares only ``finite_horizon_transfer`` with the
dataset-specific simple comparator frozen in ``strengthening_extensions.yaml``.
It does not fit or add a network-inference method.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2, norm
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/strengthening_extensions.yaml"
RESULTS = ROOT / "analyses/grn_benchmark/results"
EDGES = RESULTS / "all_candidate_edges.parquet"
REFERENCE = RESULTS / "method_results.csv"
FREEZE_MANIFEST = (
    ROOT
    / "analyses/cell_systems_expansion/strengthening_freeze_manifest.json"
)

TARGET_METRICS_OUT = RESULTS / "paired_target_metrics.parquet"
CONTRASTS_OUT = RESULTS / "paired_method_contrasts.csv"
CALIBRATION_OUT = RESULTS / "score_calibration.csv"
HETEROGENEITY_OUT = RESULTS / "dataset_heterogeneity.csv"
AUDIT_OUT = RESULTS / "paired_grn_audit.json"

TRANSFER_METHOD = "finite_horizon_transfer"
METRICS = ("auroc", "auprc", "early_precision_ratio")
CALIBRATION_BINS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    return {record["path"]: record["sha256"] for record in manifest["inputs"]}


def verify_frozen_inputs() -> dict[str, dict[str, Any]]:
    expected = frozen_hashes()
    paths = {
        "config/strengthening_extensions.yaml": CONFIG,
        "analyses/grn_benchmark/results/all_candidate_edges.parquet": EDGES,
        "analyses/grn_benchmark/results/method_results.csv": REFERENCE,
    }
    checks: dict[str, dict[str, Any]] = {}
    for relative, path in paths.items():
        observed = sha256(path)
        expected_hash = expected.get(relative)
        matches = observed == expected_hash
        checks[relative] = {
            "bytes": path.stat().st_size,
            "expected_sha256": expected_hash,
            "observed_sha256": observed,
            "matches_freeze_manifest": matches,
        }
        if not matches:
            raise RuntimeError(f"Frozen input hash mismatch: {relative}")
    return checks


def read_frozen_spec() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family = config["families"]["grn_benchmark"]
    if family["action"] != "run_targeted_extension":
        raise RuntimeError("GRN extension is not enabled by the frozen config")
    if config["correction"] != "benjamini_hochberg_within_family":
        raise RuntimeError("Unexpected multiplicity correction in frozen config")
    match = re.search(r"(\d[\d,]*)\s+candidate method-edge rows", family["eligible_denominator"])
    if match is None:
        raise RuntimeError("Could not read the frozen candidate-row denominator")
    expected_source_rows = int(match.group(1).replace(",", ""))
    match = re.search(r"(\d+)\s+replicates", family["uncertainty"])
    if match is None:
        raise RuntimeError("Could not read the frozen bootstrap count")
    bootstrap_replicates = int(match.group(1))
    return {
        "seed": int(config["seed"]),
        "alpha": float(config["alpha"]),
        "confidence_level": float(config["confidence_level"]),
        "correction": config["correction"],
        "multiplicity_family": family["multiplicity_family"],
        "comparison_by_dataset": dict(family["comparison_by_dataset"]),
        "bootstrap_replicates": bootstrap_replicates,
        "expected_source_rows": expected_source_rows,
        "minimum_targets": 100,
        "hypothesis": family["hypothesis"],
        # PyYAML's YAML 1.1 resolver treats the unquoted key ``null`` as None.
        "null": family.get("null", family.get(None)),
        "stopping_rule": family["stopping_rule"],
        "decision_criterion": family["decision_criterion"],
    }


def target_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Match the frozen benchmark metric definitions at target grain."""
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("Target metrics require both gold classes")
    order = np.argsort(scores)[::-1]
    early_precision = labels[order[:positives]].mean()
    prevalence = labels.mean()
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "early_precision_ratio": float(early_precision / prevalence),
    }


def build_paired_target_metrics(
    edges: pd.DataFrame,
    comparators: dict[str, str],
    minimum_targets: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    pairing_audit: dict[str, dict[str, Any]] = {}
    paired_edge_counts: dict[str, int] = {}

    for dataset, comparator in comparators.items():
        dataset_edges = edges.loc[edges["dataset"].eq(dataset)]
        transfer = dataset_edges.loc[
            dataset_edges["method"].eq(TRANSFER_METHOD),
            ["regulator", "target", "score", "gold"],
        ].rename(columns={"score": "transfer_score", "gold": "transfer_gold"})
        simple = dataset_edges.loc[
            dataset_edges["method"].eq(comparator),
            ["regulator", "target", "score", "gold"],
        ].rename(columns={"score": "comparator_score", "gold": "comparator_gold"})

        transfer_duplicates = int(transfer.duplicated(["regulator", "target"]).sum())
        comparator_duplicates = int(simple.duplicated(["regulator", "target"]).sum())
        if transfer_duplicates or comparator_duplicates:
            raise RuntimeError(f"Duplicate candidate edge keys in {dataset}")

        paired = transfer.merge(
            simple,
            on=["regulator", "target"],
            how="outer",
            validate="one_to_one",
            indicator=True,
            sort=False,
        )
        left_only = int(paired["_merge"].eq("left_only").sum())
        right_only = int(paired["_merge"].eq("right_only").sum())
        if left_only or right_only:
            raise RuntimeError(f"Candidate edge keys are not identical in {dataset}")
        paired = paired.drop(columns="_merge")
        gold_mismatches = int(
            paired["transfer_gold"].ne(paired["comparator_gold"]).sum()
        )
        if gold_mismatches:
            raise RuntimeError(f"Gold labels differ between methods in {dataset}")
        paired["gold"] = paired.pop("transfer_gold").astype(np.int8)
        paired = paired.drop(columns="comparator_gold")

        finite_transfer = int(np.isfinite(paired["transfer_score"]).sum())
        finite_comparator = int(np.isfinite(paired["comparator_score"]).sum())
        if finite_transfer != len(paired) or finite_comparator != len(paired):
            raise RuntimeError(f"Non-finite score in {dataset}")

        gold_by_target = paired.groupby("target", sort=True)["gold"].agg(
            ["sum", "count", "nunique"]
        )
        eligible_targets = gold_by_target.index[gold_by_target["nunique"].eq(2)]
        if len(eligible_targets) < minimum_targets:
            raise RuntimeError(
                f"{dataset} has {len(eligible_targets)} paired targets; "
                f"minimum is {minimum_targets}"
            )

        for target, block in paired.loc[paired["target"].isin(eligible_targets)].groupby(
            "target", sort=True
        ):
            labels = block["gold"].to_numpy(dtype=int)
            transfer_scores = block["transfer_score"].to_numpy(dtype=float)
            comparator_scores = block["comparator_score"].to_numpy(dtype=float)
            transfer_metrics = target_metrics(labels, transfer_scores)
            comparator_metrics = target_metrics(labels, comparator_scores)
            record: dict[str, Any] = {
                "dataset": dataset,
                "target": target,
                "comparator_method": comparator,
                "n_edges": int(len(block)),
                "gold_edges": int(labels.sum()),
                "non_gold_edges": int(len(labels) - labels.sum()),
            }
            for metric in METRICS:
                transfer_value = transfer_metrics[metric]
                comparator_value = comparator_metrics[metric]
                record[f"{TRANSFER_METHOD}_{metric}"] = transfer_value
                record[f"comparator_{metric}"] = comparator_value
                record[f"difference_{metric}"] = transfer_value - comparator_value
            records.append(record)

        paired_edge_counts[dataset] = int(len(paired))
        pairing_audit[dataset] = {
            "comparator_method": comparator,
            "transfer_rows": int(len(transfer)),
            "comparator_rows": int(len(simple)),
            "matched_edge_pairs": int(len(paired)),
            "transfer_duplicate_edge_keys": transfer_duplicates,
            "comparator_duplicate_edge_keys": comparator_duplicates,
            "unmatched_transfer_edges": left_only,
            "unmatched_comparator_edges": right_only,
            "gold_label_mismatches": gold_mismatches,
            "edge_keys_identical": left_only == 0 and right_only == 0,
            "gold_labels_identical": gold_mismatches == 0,
            "finite_transfer_scores": finite_transfer,
            "finite_comparator_scores": finite_comparator,
            "all_scores_finite": finite_transfer == len(paired)
            and finite_comparator == len(paired),
            "all_targets": int(len(gold_by_target)),
            "evaluable_targets": int(len(eligible_targets)),
            "single_class_targets_excluded": int(
                len(gold_by_target) - len(eligible_targets)
            ),
            "minimum_targets": int(minimum_targets),
            "minimum_power_passed": bool(len(eligible_targets) >= minimum_targets),
            "positive_edges": int(paired["gold"].sum()),
            "negative_edges": int(len(paired) - paired["gold"].sum()),
        }

    table = pd.DataFrame.from_records(records).sort_values(
        ["dataset", "target"], kind="stable", ignore_index=True
    )
    if table.duplicated(["dataset", "target"]).any():
        raise RuntimeError("Paired target metrics are not unique by dataset and target")
    return table, pairing_audit, paired_edge_counts


def check_reference_reproduction(
    target_table: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    errors: list[float] = []
    checks = 0
    for (dataset, comparator), block in target_table.groupby(
        ["dataset", "comparator_method"], sort=True
    ):
        for method, prefix in (
            (TRANSFER_METHOD, TRANSFER_METHOD),
            (comparator, "comparator"),
        ):
            row = reference.loc[
                reference["dataset"].eq(dataset) & reference["method"].eq(method)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing unique frozen reference for {dataset}/{method}")
            if int(row.iloc[0]["bootstrap_targets"]) != len(block):
                raise RuntimeError(f"Reference target count mismatch for {dataset}/{method}")
            for metric in METRICS:
                observed = float(block[f"{prefix}_{metric}"].mean())
                expected = float(row.iloc[0][f"target_macro_{metric}"])
                errors.append(abs(observed - expected))
                checks += 1
    maximum_error = max(errors, default=np.nan)
    if not np.isfinite(maximum_error) or maximum_error > 1e-12:
        raise RuntimeError(
            f"Independent target-macro reproduction failed (max error {maximum_error})"
        )
    return {
        "statistics_checked": checks,
        "maximum_absolute_error": float(maximum_error),
        "tolerance": 1e-12,
        "passed": True,
    }


def bootstrap_contrasts(
    target_table: pd.DataFrame,
    comparators: dict[str, str],
    seed: int,
    replicates: int,
    confidence_level: float,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[dict[str, Any]] = []
    seed_offsets: dict[str, int] = {}
    quantile_low = (1.0 - confidence_level) / 2.0
    quantile_high = 1.0 - quantile_low

    for dataset_index, (dataset, comparator) in enumerate(comparators.items()):
        block = target_table.loc[target_table["dataset"].eq(dataset)]
        for metric_index, metric in enumerate(METRICS):
            difference = block[f"difference_{metric}"].to_numpy(dtype=float)
            observed = float(difference.mean())
            bootstrap_seed = seed + dataset_index * 100 + metric_index
            seed_offsets[f"{dataset}/{metric}"] = bootstrap_seed - seed
            rng = np.random.default_rng(bootstrap_seed)
            draw = rng.integers(
                0, len(difference), size=(replicates, len(difference))
            )
            bootstrap = difference[draw].mean(axis=1)
            ci_low, ci_high = np.quantile(
                bootstrap, [quantile_low, quantile_high]
            )
            centered_null = bootstrap - observed
            p_value = (
                np.count_nonzero(np.abs(centered_null) >= abs(observed)) + 1
            ) / (replicates + 1)
            records.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "method": TRANSFER_METHOD,
                    "comparator_method": comparator,
                    "n_targets": int(len(block)),
                    "mean_method_metric": float(
                        block[f"{TRANSFER_METHOD}_{metric}"].mean()
                    ),
                    "mean_comparator_metric": float(
                        block[f"comparator_{metric}"].mean()
                    ),
                    "mean_paired_difference": observed,
                    "ci_level": confidence_level,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_standard_error": float(bootstrap.std(ddof=1)),
                    "p_value": float(p_value),
                    "multiplicity_family": "grn_paired_metrics",
                }
            )

    contrasts = pd.DataFrame.from_records(records)
    if len(contrasts) != len(comparators) * len(METRICS):
        raise RuntimeError("Unexpected number of paired contrasts")
    contrasts["q_value"] = benjamini_hochberg(
        contrasts["p_value"].to_numpy(dtype=float)
    )
    transfer_superior = contrasts["q_value"].lt(alpha) & contrasts["ci_low"].gt(0)
    comparator_superior = contrasts["q_value"].lt(alpha) & contrasts["ci_high"].lt(0)
    contrasts["corrected_result"] = np.select(
        [transfer_superior, comparator_superior],
        [
            "finite_horizon_transfer_superior",
            "frozen_simple_comparator_superior",
        ],
        default="unresolved",
    )
    contrasts["reject_null_after_bh"] = contrasts["q_value"].lt(alpha)
    contrasts = contrasts.sort_values(
        ["dataset", "metric"], kind="stable", ignore_index=True
    )
    return contrasts, seed_offsets


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("BH correction requires finite p-values in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def held_out_rank_calibration(
    edges: pd.DataFrame,
    comparators: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset, comparator in comparators.items():
        for method in (TRANSFER_METHOD, comparator):
            table = edges.loc[
                edges["dataset"].eq(dataset) & edges["method"].eq(method),
                ["regulator", "target", "score", "gold"],
            ].copy()
            if table.empty:
                raise RuntimeError(f"No calibration edges for {dataset}/{method}")
            if not np.isfinite(table["score"]).all():
                raise RuntimeError(f"Non-finite calibration score for {dataset}/{method}")

            # Rank assignment is outcome-free. Regulator and target break score ties
            # deterministically; bins therefore have near-equal edge counts.
            ordering = table.sort_values(
                ["score", "regulator", "target"], kind="stable"
            ).index.to_numpy()
            ranks = pd.Series(np.arange(len(table), dtype=np.int64), index=ordering)
            table["score_rank"] = (ranks.reindex(table.index).to_numpy() + 0.5) / len(
                table
            )
            table["rank_bin"] = np.minimum(
                (ranks.reindex(table.index).to_numpy() * CALIBRATION_BINS)
                // len(table),
                CALIBRATION_BINS - 1,
            ).astype(np.int8)

            bin_group = table.groupby("rank_bin", sort=False)["gold"]
            bin_count = bin_group.transform("size").to_numpy(dtype=np.int64)
            bin_positive = bin_group.transform("sum").to_numpy(dtype=np.int64)
            target_bin_group = table.groupby(
                ["rank_bin", "target"], sort=False
            )["gold"]
            target_bin_count = target_bin_group.transform("size").to_numpy(
                dtype=np.int64
            )
            target_bin_positive = target_bin_group.transform("sum").to_numpy(
                dtype=np.int64
            )
            training_count = bin_count - target_bin_count
            training_positive = bin_positive - target_bin_positive
            if (training_count <= 0).any():
                raise RuntimeError(
                    f"Empty target-held-out calibration training bin in {dataset}/{method}"
                )
            table["held_out_probability"] = training_positive / training_count
            squared_error = np.square(
                table["held_out_probability"].to_numpy(dtype=float)
                - table["gold"].to_numpy(dtype=float)
            )
            table["squared_error"] = squared_error
            brier = float(squared_error.mean())

            bin_rows: list[dict[str, Any]] = []
            for rank_bin, block in table.groupby("rank_bin", sort=True):
                mean_probability = float(block["held_out_probability"].mean())
                observed_frequency = float(block["gold"].mean())
                bin_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "rank_bin": int(rank_bin) + 1,
                        "n_edges": int(len(block)),
                        "n_targets": int(block["target"].nunique()),
                        "gold_edges": int(block["gold"].sum()),
                        "mean_score_rank": float(block["score_rank"].mean()),
                        "minimum_score": float(block["score"].min()),
                        "maximum_score": float(block["score"].max()),
                        "mean_held_out_probability": mean_probability,
                        "observed_gold_frequency": observed_frequency,
                        "absolute_calibration_gap": abs(
                            mean_probability - observed_frequency
                        ),
                        "bin_brier_score": float(block["squared_error"].mean()),
                    }
                )
            total = sum(row["n_edges"] for row in bin_rows)
            ece = sum(
                row["n_edges"] / total * row["absolute_calibration_gap"]
                for row in bin_rows
            )
            for row in bin_rows:
                row["brier_score"] = brier
                row["expected_calibration_error"] = float(ece)
                row["calibration_bins"] = CALIBRATION_BINS
                row["calibration_estimator"] = (
                    "empirical_probability_by_global_score_rank_decile_"
                    "excluding_the_evaluated_target"
                )
                records.append(row)

    calibration = pd.DataFrame.from_records(records).sort_values(
        ["dataset", "method", "rank_bin"], kind="stable", ignore_index=True
    )
    expected_rows = len(comparators) * 2 * CALIBRATION_BINS
    if len(calibration) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} calibration rows, observed {len(calibration)}"
        )
    return calibration


def random_effects_heterogeneity(contrasts: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric in METRICS:
        metric_contrasts = contrasts.loc[contrasts["metric"].eq(metric)].sort_values(
            "dataset", kind="stable"
        )
        effects: list[float] = []
        variances: list[float] = []
        details: dict[str, dict[str, float | int]] = {}
        for row in metric_contrasts.itertuples(index=False):
            dataset = str(row.dataset)
            effect = float(row.mean_paired_difference)
            standard_error = float(row.bootstrap_standard_error)
            if not np.isfinite(standard_error) or standard_error <= 0:
                raise RuntimeError(
                    f"Invalid bootstrap standard error for {dataset}/{metric}"
                )
            variance = max(standard_error**2, np.finfo(float).tiny)
            effects.append(effect)
            variances.append(variance)
            details[dataset] = {
                "effect": effect,
                "standard_error": standard_error,
                "targets": int(row.n_targets),
            }

        y = np.asarray(effects, dtype=float)
        variance = np.asarray(variances, dtype=float)
        fixed_weight = 1.0 / variance
        fixed_effect = float(np.sum(fixed_weight * y) / np.sum(fixed_weight))
        q = float(np.sum(fixed_weight * np.square(y - fixed_effect)))
        degrees_freedom = len(y) - 1
        c_value = float(
            np.sum(fixed_weight)
            - np.sum(np.square(fixed_weight)) / np.sum(fixed_weight)
        )
        tau_squared = max(0.0, (q - degrees_freedom) / c_value)
        random_weight = 1.0 / (variance + tau_squared)
        random_effect = float(np.sum(random_weight * y) / np.sum(random_weight))
        random_effect_se = float(np.sqrt(1.0 / np.sum(random_weight)))
        ci_low = random_effect - norm.ppf(0.975) * random_effect_se
        ci_high = random_effect + norm.ppf(0.975) * random_effect_se
        prediction_se = float(np.sqrt(tau_squared + random_effect_se**2))
        prediction_low = random_effect - norm.ppf(0.975) * prediction_se
        prediction_high = random_effect + norm.ppf(0.975) * prediction_se
        i_squared = max(0.0, (q - degrees_freedom) / q) * 100 if q > 0 else 0.0
        records.append(
            {
                "metric": metric,
                "method": TRANSFER_METHOD,
                "comparator_definition": "frozen_dataset_specific_simple_method",
                "k_datasets": int(len(y)),
                "fixed_effect": fixed_effect,
                "random_effect": random_effect,
                "random_effect_standard_error": random_effect_se,
                "random_effect_ci_low": float(ci_low),
                "random_effect_ci_high": float(ci_high),
                "random_effect_p_value": float(
                    2 * norm.sf(abs(random_effect / random_effect_se))
                ),
                "heterogeneity_q": q,
                "heterogeneity_degrees_freedom": int(degrees_freedom),
                "heterogeneity_p_value": float(chi2.sf(q, degrees_freedom)),
                "tau_squared": float(tau_squared),
                "i_squared_percent": float(i_squared),
                "prediction_interval_low": float(prediction_low),
                "prediction_interval_high": float(prediction_high),
                "tau_squared_estimator": "DerSimonian-Laird",
                "dataset_effects_json": json.dumps(details, sort_keys=True),
            }
        )
    return pd.DataFrame.from_records(records)


def transfer_superiority_decisions(
    contrasts: pd.DataFrame,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for metric, block in contrasts.groupby("metric", sort=True):
        transfer_count = int(
            block["corrected_result"].eq("finite_horizon_transfer_superior").sum()
        )
        comparator_count = int(
            block["corrected_result"].eq("frozen_simple_comparator_superior").sum()
        )
        criterion_met = transfer_count >= 3 and comparator_count == 0
        decisions[metric] = {
            "datasets": int(len(block)),
            "finite_horizon_transfer_superior_datasets": transfer_count,
            "frozen_simple_comparator_superior_datasets": comparator_count,
            "unresolved_datasets": int(
                block["corrected_result"].eq("unresolved").sum()
            ),
            "finite_horizon_transfer_superiority_criterion_met": criterion_met,
            "status": "supported" if criterion_met else "failed",
            "criterion": (
                f"q < {alpha:g} and lower CI > 0 in at least 3 of 4 datasets, "
                "with no corrected opposite effects"
            ),
        }
    return decisions


def write_csv(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def write_parquet(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False, engine="pyarrow", compression="snappy")
    temporary.replace(path)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frozen_input_checks = verify_frozen_inputs()
    specification = read_frozen_spec()
    comparators = specification["comparison_by_dataset"]
    if "mESC" in comparators:
        raise RuntimeError("mESC must remain unavailable and cannot be relaxed")
    if specification["bootstrap_replicates"] != 2000:
        raise RuntimeError("Frozen GRN bootstrap count is not 2,000")

    edges = pd.read_parquet(EDGES)
    required_columns = {"dataset", "method", "regulator", "target", "score", "gold"}
    missing_columns = required_columns.difference(edges.columns)
    if missing_columns:
        raise RuntimeError(f"Candidate-edge table is missing columns: {missing_columns}")
    if len(edges) != specification["expected_source_rows"]:
        raise RuntimeError(
            f"Frozen candidate row count changed: {len(edges)} != "
            f"{specification['expected_source_rows']}"
        )
    if not edges["gold"].isin([0, 1]).all():
        raise RuntimeError("Gold labels must be binary")
    source_duplicate_keys = int(
        edges.duplicated(["dataset", "method", "regulator", "target"]).sum()
    )
    if source_duplicate_keys:
        raise RuntimeError("Frozen candidate-edge table contains duplicate keys")

    selected_mask = pd.Series(False, index=edges.index)
    for dataset, comparator in comparators.items():
        selected_mask |= edges["dataset"].eq(dataset) & edges["method"].isin(
            [TRANSFER_METHOD, comparator]
        )
    selected = edges.loc[selected_mask].copy()
    target_table, pairing_audit, paired_edge_counts = build_paired_target_metrics(
        selected, comparators, specification["minimum_targets"]
    )
    reference = pd.read_csv(REFERENCE)
    reproduction = check_reference_reproduction(target_table, reference)
    contrasts, seed_offsets = bootstrap_contrasts(
        target_table,
        comparators,
        specification["seed"],
        specification["bootstrap_replicates"],
        specification["confidence_level"],
        specification["alpha"],
    )
    calibration = held_out_rank_calibration(selected, comparators)
    heterogeneity = random_effects_heterogeneity(contrasts)

    if len(contrasts) != 12:
        raise RuntimeError("The frozen BH family must contain exactly 12 contrasts")
    if contrasts["multiplicity_family"].nunique() != 1:
        raise RuntimeError("Paired contrasts escaped the frozen multiplicity family")
    if len(heterogeneity) != len(METRICS):
        raise RuntimeError("Expected one heterogeneity synthesis per metric")

    write_parquet(target_table, TARGET_METRICS_OUT)
    write_csv(contrasts, CONTRASTS_OUT)
    write_csv(calibration, CALIBRATION_OUT)
    write_csv(heterogeneity, HETEROGENEITY_OUT)

    # Round-trip checks protect the actual artifacts, not just in-memory objects.
    round_trip_target = pd.read_parquet(TARGET_METRICS_OUT)
    round_trip_contrasts = pd.read_csv(CONTRASTS_OUT)
    round_trip_calibration = pd.read_csv(CALIBRATION_OUT)
    round_trip_heterogeneity = pd.read_csv(HETEROGENEITY_OUT)
    if len(round_trip_target) != len(target_table):
        raise RuntimeError("Target-metric parquet row-count round trip failed")
    if len(round_trip_contrasts) != 12:
        raise RuntimeError("Contrast CSV row-count round trip failed")
    if len(round_trip_calibration) != len(comparators) * 2 * CALIBRATION_BINS:
        raise RuntimeError("Calibration CSV row-count round trip failed")
    if len(round_trip_heterogeneity) != len(METRICS):
        raise RuntimeError("Heterogeneity CSV row-count round trip failed")

    decisions = transfer_superiority_decisions(
        contrasts, specification["alpha"]
    )
    audit = {
        "status": "completed",
        "analysis": "frozen_paired_grn_benchmark_extension",
        "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "frozen_input_checks": frozen_input_checks,
        "frozen_specification": {
            "hypothesis": specification["hypothesis"],
            "null": specification["null"],
            "comparison_by_dataset": comparators,
            "seed": specification["seed"],
            "bootstrap_replicates": specification["bootstrap_replicates"],
            "confidence_level": specification["confidence_level"],
            "alpha": specification["alpha"],
            "correction": specification["correction"],
            "multiplicity_family": specification["multiplicity_family"],
            "minimum_targets": specification["minimum_targets"],
            "stopping_rule": specification["stopping_rule"],
            "decision_criterion": specification["decision_criterion"],
            "bootstrap_seed_offsets": seed_offsets,
            "bootstrap_p_value_method": (
                "two-sided absolute observed mean against the bootstrap distribution "
                "centered on the paired-difference null, with a plus-one correction"
            ),
        },
        "availability": {
            "estimated_datasets": list(comparators),
            "mESC": {
                "status": "unavailable",
                "candidate_rows_in_frozen_edge_table": int(
                    edges["dataset"].eq("mESC").sum()
                ),
                "relaxation_attempted": False,
                "reason": (
                    "No eligible regulators passed the frozen selection rule; "
                    "mESC has no rows in the frozen candidate-edge table."
                ),
            },
        },
        "row_counts": {
            "source_candidate_method_edges": int(len(edges)),
            "selected_candidate_method_edges": int(len(selected)),
            "paired_edge_keys": int(sum(paired_edge_counts.values())),
            "paired_target_metrics": int(len(target_table)),
            "paired_method_contrasts": int(len(contrasts)),
            "score_calibration_bins": int(len(calibration)),
            "dataset_heterogeneity_metrics": int(len(heterogeneity)),
        },
        "source_duplicate_keys": source_duplicate_keys,
        "pairing_checks": pairing_audit,
        "independent_reference_reproduction": reproduction,
        "multiplicity_check": {
            "family": specification["multiplicity_family"],
            "contrasts": int(len(contrasts)),
            "expected_contrasts": 12,
            "correction": specification["correction"],
            "passed": len(contrasts) == 12,
        },
        "calibration": {
            "methods": int(calibration.groupby(["dataset", "method"]).ngroups),
            "bins_per_method": CALIBRATION_BINS,
            "target_outcome_leakage": False,
            "rank_definition": (
                "Outcome-free global within-dataset/method score ranks; score ties "
                "are broken deterministically by regulator and target."
            ),
            "probability_definition": (
                "Empirical gold-edge frequency in the score-rank decile after "
                "removing every edge for the evaluated target."
            ),
            "metrics": ["Brier score", "expected calibration error"],
        },
        "heterogeneity": {
            "datasets": len(comparators),
            "metrics": list(METRICS),
            "estimator": "DerSimonian-Laird random effects",
            "within_dataset_variance": (
                "Squared standard deviation across the 2,000 paired target-bootstrap means"
            ),
            "included_in_12_contrast_bh_family": False,
        },
        "decisions": {
            "finite_horizon_transfer_superiority_by_metric": decisions,
            "frozen_conclusion_preserved": not any(
                result["finite_horizon_transfer_superiority_criterion_met"]
                for result in decisions.values()
            ),
            "new_methods_added": False,
        },
        "output_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (
                TARGET_METRICS_OUT,
                CONTRASTS_OUT,
                CALIBRATION_OUT,
                HETEROGENEITY_OUT,
            )
        },
        "limitations": [
            (
                "Target-macro metrics exclude targets whose frozen candidate edges "
                "contain only one gold class."
            ),
            (
                "Early precision retains the frozen score-order definition; tied "
                "scores can make the top-k membership depend on deterministic row order."
            ),
            (
                "Rank-decile calibration is descriptive and target-held-out, but it "
                "does not establish transportable absolute probabilities across datasets."
            ),
            (
                "Brier scores are strongly influenced by the low gold-edge prevalence; "
                "ECE and the ten calibration bins are reported alongside them."
            ),
            (
                "Random-effects heterogeneity uses only four datasets; tau-squared, "
                "I-squared, normal intervals, and prediction intervals are imprecise."
            ),
            (
                "The BH family contains only the 12 prespecified dataset-metric paired "
                "contrasts; heterogeneity synthesis p-values are descriptive and unadjusted."
            ),
            "mESC remains unavailable and no eligibility rule was relaxed.",
        ],
    }
    temporary_audit = AUDIT_OUT.with_suffix(AUDIT_OUT.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_audit.replace(AUDIT_OUT)
    print(json.dumps(audit["row_counts"], sort_keys=True))
    print(json.dumps(audit["decisions"], sort_keys=True))


if __name__ == "__main__":
    main()
