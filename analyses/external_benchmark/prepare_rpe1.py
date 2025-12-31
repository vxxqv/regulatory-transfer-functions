"""Prepare target-level RPE1 holdout outcomes under the frozen rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]


def bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def moments(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum((second - mean**2) * n / max(n - 1, 1), 0.0)
    return mean, variance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad", type=Path, default=ROOT / "work/replogle22rpe1_processed_complete.h5ad"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "analyses/external_benchmark/rpe1_prepared"
    )
    parser.add_argument("--minimum-cells", type=int, default=30)
    parser.add_argument("--minimum-absolute-difference", type=float, default=0.10)
    parser.add_argument("--fdr", type=float, default=0.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(args.h5ad)
    matrix = sparse.csr_matrix(data.X, dtype=np.float64)
    labels = data.obs["perturbation"].astype(str).to_numpy()
    genes = data.var_names.astype(str).to_numpy()
    controls = labels == "control"
    if controls.sum() < 500:
        raise ValueError("Frozen minimum of 500 control cells was not met")
    control_mean, control_variance = moments(matrix[controls])
    control_n = int(controls.sum())

    cd4_targets = set(
        pd.read_csv(ROOT / "analyses/external_benchmark/target_folds.tsv", sep="\t")[
            "target_gene"
        ].astype(str)
    )
    fold_map = pd.read_csv(ROOT / "analyses/external_benchmark/target_folds.tsv", sep="\t").set_index(
        "target_gene"
    )["fold"]
    counts = pd.Series(labels[~controls]).value_counts().sort_index()
    eligible = counts[(counts >= args.minimum_cells) & counts.index.isin(cd4_targets)]
    records: list[dict[str, object]] = []
    effect_rows: list[sparse.csr_matrix] = []
    gene_index = {gene: index for index, gene in enumerate(genes)}

    for position, (target, n_cells) in enumerate(eligible.items(), start=1):
        selected = labels == target
        block = matrix[selected]
        target_mean, target_variance = moments(block)
        difference = target_mean - control_mean
        standard_error_squared = target_variance / n_cells + control_variance / control_n
        standard_error = np.sqrt(np.maximum(standard_error_squared, np.finfo(float).tiny))
        statistic = difference / standard_error
        numerator = standard_error_squared**2
        denominator = (
            (target_variance / n_cells) ** 2 / max(n_cells - 1, 1)
            + (control_variance / control_n) ** 2 / max(control_n - 1, 1)
        )
        degrees_freedom = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.inf), where=denominator > 0
        )
        p_value = 2.0 * student_t.sf(np.abs(statistic), degrees_freedom)
        q_value = bh(np.nan_to_num(p_value, nan=1.0, posinf=1.0, neginf=1.0))
        significant = (q_value < args.fdr) & (np.abs(difference) >= args.minimum_absolute_difference)
        response = difference.copy()
        response[~significant] = 0.0
        response_index = np.flatnonzero(response)
        effect_rows.append(
            sparse.csr_matrix(
                (response[response_index], (np.zeros(len(response_index)), response_index)),
                shape=(1, len(genes)),
            )
        )
        on_target_index = gene_index.get(str(target))
        records.append(
            {
                "target_gene": target,
                "fold": int(fold_map.loc[target]),
                "n_cells": int(n_cells),
                "control_cells": control_n,
                "control_target_expression": float(control_mean[on_target_index])
                if on_target_index is not None
                else np.nan,
                "on_target_mean_difference": float(difference[on_target_index])
                if on_target_index is not None
                else np.nan,
                "n_differentially_expressed_genes": int(significant.sum()),
                "log1p_differentially_expressed_genes": float(np.log1p(significant.sum())),
                "response_l2_norm": float(np.linalg.norm(response)),
                "detected_response_genes": int(np.count_nonzero(response)),
            }
        )
        if position % 100 == 0:
            print(f"processed {position}/{len(eligible)} targets", flush=True)

    effects = sparse.vstack(effect_rows, format="csr") if effect_rows else sparse.csr_matrix((0, len(genes)))
    target_table = pd.DataFrame(records)
    target_table.to_parquet(args.output / "eligible_targets.parquet", index=False)
    sparse.save_npz(args.output / "signed_significant_effects.npz", effects)
    pd.DataFrame({"gene": genes}).to_parquet(args.output / "genes.parquet", index=False)

    exclusions = []
    for target, n_cells in counts.items():
        reasons = []
        if n_cells < args.minimum_cells:
            reasons.append("fewer_than_30_cells")
        if target not in cd4_targets:
            reasons.append("not_measured_in_frozen_cd4_target_set")
        if not reasons:
            continue
        exclusions.append({"target_gene": target, "n_cells": int(n_cells), "reason": ";".join(reasons)})
    pd.DataFrame(exclusions).to_csv(args.output / "excluded_targets.csv", index=False)
    audit = {
        "cells": int(data.n_obs),
        "genes": int(data.n_vars),
        "control_cells": control_n,
        "perturbation_targets_total": int(len(counts)),
        "eligible_shared_targets": int(len(target_table)),
        "excluded_targets": int(len(exclusions)),
        "effect_nonzero_entries": int(effects.nnz),
        "minimum_cells": args.minimum_cells,
        "fdr": args.fdr,
        "minimum_absolute_mean_difference": args.minimum_absolute_difference,
    }
    (args.output / "preparation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
