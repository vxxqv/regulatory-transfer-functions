from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "buffering_deletion_extension.yaml"
RESULTS = ROOT / "analyses" / "primary" / "results"
INPUT = RESULTS / "transfer_phenotypes.parquet"
STATES = ["Rest", "Stim8hr", "Stim48hr"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def select_rows(residual: np.ndarray, tail: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    selected, outcomes = [], []
    for heldout in range(residual.shape[1]):
        training = np.delete(residual, heldout, axis=1).mean(axis=1)
        threshold = np.quantile(training, tail)
        indices = np.flatnonzero(training <= threshold)
        selected.append(indices)
        outcomes.append(residual[indices, heldout])
    return selected, outcomes


def estimates(residual: np.ndarray, tail: float) -> np.ndarray:
    _, outcomes = select_rows(residual, tail)
    state_values = [values.mean() for values in outcomes]
    return np.asarray(state_values + [np.concatenate(outcomes).mean()])


def matched_strata(values: np.ndarray) -> np.ndarray:
    first = pd.qcut(pd.Series(values[:, 0]).rank(method="first"), 5, labels=False).to_numpy()
    second = pd.qcut(pd.Series(values[:, 1]).rank(method="first"), 5, labels=False).to_numpy()
    return first * 5 + second


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if sha256(INPUT) != config["input_sha256"]:
        raise RuntimeError("Transfer phenotype input hash does not match the frozen configuration")
    frame = pd.read_parquet(INPUT)
    required = ["target_contrast", "culture_condition", "transfer_residual", "log1p_n_downstream", "log1p_target_baseMean"]
    frame = frame[required].dropna()
    if frame.duplicated(["target_contrast", "culture_condition"]).any():
        raise RuntimeError("Duplicate target-state rows are not permitted")
    residual = frame.pivot(index="target_contrast", columns="culture_condition", values="transfer_residual").reindex(columns=STATES).dropna()
    targets = residual.index
    degree = frame.pivot(index="target_contrast", columns="culture_condition", values="log1p_n_downstream").reindex(index=targets, columns=STATES)
    expression = frame.pivot(index="target_contrast", columns="culture_condition", values="log1p_target_baseMean").reindex(index=targets, columns=STATES)
    if degree.isna().any().any() or expression.isna().any().any():
        raise RuntimeError("Complete-state matching covariates are not finite")
    residual_values = residual.to_numpy(dtype=float)
    degree_values = degree.to_numpy(dtype=float)
    expression_values = expression.to_numpy(dtype=float)
    tail = float(config["tail_fraction"])
    selected, selected_outcomes = select_rows(residual_values, tail)
    point = estimates(residual_values, tail)
    selected_counts = [len(indices) for indices in selected] + [sum(len(indices) for indices in selected)]
    if selected_counts[-1] < 100 or min(selected_counts[:3]) < 30:
        raise RuntimeError("Buffering deletion minimum-power gate failed")

    strata = []
    for heldout in range(len(STATES)):
        training_degree = np.delete(degree_values, heldout, axis=1).mean(axis=1)
        training_expression = np.delete(expression_values, heldout, axis=1).mean(axis=1)
        strata.append(matched_strata(np.column_stack([training_degree, training_expression])))

    rng = np.random.default_rng(int(config["seed"]))
    bootstrap = np.empty((2000, 4), dtype=float)
    for replicate in range(2000):
        sampled = rng.integers(0, len(targets), len(targets))
        bootstrap[replicate] = estimates(residual_values[sampled], tail)

    null = np.empty((2000, 4), dtype=float)
    for replicate in range(2000):
        state_values = []
        for heldout in range(len(STATES)):
            permuted = residual_values[:, heldout].copy()
            for stratum in np.unique(strata[heldout]):
                indices = np.flatnonzero(strata[heldout] == stratum)
                permuted[indices] = rng.permutation(permuted[indices])
            state_values.append(permuted[selected[heldout]])
        null[replicate, :3] = [values.mean() for values in state_values]
        null[replicate, 3] = np.concatenate(state_values).mean()

    labels = STATES + ["Pooled"]
    records = []
    for index, label in enumerate(labels):
        records.append(
            {
                "heldout_state": label,
                "estimate": point[index],
                "ci_low": float(np.quantile(bootstrap[:, index], 0.025)),
                "ci_high": float(np.quantile(bootstrap[:, index], 0.975)),
                "p_lower_tail": float((1 + np.sum(null[:, index] <= point[index])) / 2001),
                "p_upper_tail": float((1 + np.sum(null[:, index] >= point[index])) / 2001),
                "selected_rows": selected_counts[index],
                "eligible_targets": len(targets),
                "bootstrap_replicates": 2000,
                "permutation_replicates": 2000,
            }
        )
    summary = pd.DataFrame(records)
    summary["q_lower_tail"] = bh(summary["p_lower_tail"].to_numpy())
    summary["q_upper_tail"] = bh(summary["p_upper_tail"].to_numpy())
    summary["decision"] = np.where(
        (summary["ci_high"] < 0) & (summary["q_lower_tail"] < 0.05),
        "supported",
        "unresolved",
    )

    influence_records = []
    all_indices = np.arange(len(targets))
    for index, target in enumerate(targets):
        retained = all_indices != index
        estimate = estimates(residual_values[retained], tail)[3]
        influence_records.append(
            {
                "target": target,
                "leave_one_target_estimate": estimate,
                "change_from_full": estimate - point[3],
            }
        )
    influence = pd.DataFrame(influence_records).sort_values("change_from_full", key=np.abs, ascending=False)
    unavailable = pd.DataFrame(
        [
            ["leave_one_guide", "unavailable", "Aggregate target-state rows do not contain guide-resolved residuals."],
            ["leave_one_donor", "unavailable", "Aggregate target-state rows do not contain donor-resolved residuals."],
            ["leave_one_locus", "not_applicable", "Buffering is not a locus-level estimand."],
        ],
        columns=["analysis", "status", "reason"],
    )
    null_table = pd.DataFrame(null, columns=labels).assign(replicate=np.arange(2000)).melt(id_vars="replicate", var_name="heldout_state", value_name="estimate")
    summary.to_csv(RESULTS / "buffering_deletion_summary.csv", index=False)
    influence.to_csv(RESULTS / "buffering_leave_one_target.csv", index=False)
    unavailable.to_csv(RESULTS / "buffering_unavailable_deletions.csv", index=False)
    null_table.to_parquet(RESULTS / "buffering_deletion_null.parquet", index=False)
    audit = {
        "analysis": "frozen_buffering_deletion_extension",
        "eligible_targets": int(len(targets)),
        "selected_rows": dict(zip(labels, selected_counts, strict=True)),
        "tail_fraction": tail,
        "states": STATES,
        "config_sha256": sha256(CONFIG),
        "input_sha256": sha256(INPUT),
        "maximum_absolute_leave_one_target_change": float(influence["change_from_full"].abs().max()),
        "complete_bootstrap_replicates": bool(np.isfinite(bootstrap).all()),
        "complete_permutation_replicates": bool(np.isfinite(null).all()),
    }
    (RESULTS / "buffering_deletion_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
