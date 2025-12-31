"""Run signed vector, network gain, and contextual transfer analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=Path("data/interim/gwt_vectors"))
    parser.add_argument("--output", type=Path, default=Path("analyses/vectors/results"))
    parser.add_argument("--config", type=Path, default=Path("config/analysis.yaml"))
    parser.add_argument("--components", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.models.network_transfer import (
        aligned_state_pairs,
        cross_fitted_modules,
        js_divergence,
        module_energy,
        row_cosine,
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["study"]["seed"])
    folds = int(config["transfer_phenotypes"]["cross_validation"]["folds"])
    states = tuple(config["primary_resource"]["conditions"])
    args.output.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(args.vectors / "rows.parquet").reset_index(drop=True)
    matrix = sparse.load_npz(args.vectors / "normalized_significant_logfc.npz").tocsr()
    if matrix.shape[0] != len(rows):
        raise ValueError("Response matrix and row metadata disagree")

    oof_scores, reconstruction_cosine, folds_table = cross_fitted_modules(
        matrix,
        rows["target_contrast"].astype(str).to_numpy(),
        n_components=args.components,
        folds=folds,
        seed=seed,
    )
    reference_model = TruncatedSVD(n_components=args.components, random_state=seed)
    scores = reference_model.fit_transform(matrix)
    energies = module_energy(scores)
    dominant = np.argmax(energies, axis=1)
    row_metrics = rows.copy()
    row_metrics["response_norm"] = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).ravel()
    row_metrics["module_energy_captured"] = np.square(scores).sum(axis=1)
    row_metrics["oof_module_energy_captured"] = np.square(oof_scores).sum(axis=1)
    row_metrics["dominant_module"] = dominant + 1
    row_metrics["dominant_module_fraction"] = energies[np.arange(len(rows)), dominant]
    row_metrics["oof_reconstruction_cosine"] = reconstruction_cosine
    row_metrics.to_parquet(args.output / "network_gain_rows.parquet", index=False)
    folds_table.to_csv(args.output / "module_folds.csv", index=False)

    genes = pd.read_parquet(args.vectors / "genes.parquet")
    loading_records = []
    for component, loadings in enumerate(reference_model.components_, start=1):
        loading_records.append(
            pd.DataFrame(
                {
                    "feature_id": genes["feature_id"].astype(str),
                    "gene_name": genes["gene_name"].astype(str),
                    "module": component,
                    "loading": loadings,
                }
            )
        )
    pd.concat(loading_records, ignore_index=True).to_parquet(
        args.output / "reference_module_loadings.parquet", index=False
    )

    tensor_records: list[pd.DataFrame] = []
    for component in range(args.components):
        tensor_records.append(
            pd.DataFrame(
                {
                    "target_contrast": rows["target_contrast"].astype(str),
                    "target_gene": rows["target_contrast_gene_name"].astype(str),
                    "condition": rows["culture_condition"].astype(str),
                    "module": component + 1,
                    "score": scores[:, component],
                    "energy_fraction": energies[:, component],
                }
            )
        )
    tensor = pd.concat(tensor_records, ignore_index=True)
    tensor.to_parquet(args.output / "contextual_transfer_tensor.parquet", index=False)

    pairs = aligned_state_pairs(rows, states)
    left_index = pairs["left_index"].to_numpy(dtype=int)
    right_index = pairs["right_index"].to_numpy(dtype=int)
    left_norm = row_metrics.loc[left_index, "response_norm"].to_numpy()
    right_norm = row_metrics.loc[right_index, "response_norm"].to_numpy()
    pair_evaluable = (left_norm > 0) & (right_norm > 0)
    pairs["left_has_response"] = left_norm > 0
    pairs["right_has_response"] = right_norm > 0
    pairs["pair_evaluable"] = pair_evaluable
    response_cosine = row_cosine(matrix[left_index], matrix[right_index])
    response_cosine[~pair_evaluable] = np.nan
    pairs["response_cosine"] = response_cosine
    module_js = js_divergence(energies[left_index], energies[right_index])
    module_js[~pair_evaluable] = np.nan
    pairs["module_js_divergence"] = module_js
    gain_ratio = np.full(len(pairs), np.nan, dtype=float)
    gain_ratio[pair_evaluable] = np.abs(np.log2(right_norm[pair_evaluable] / left_norm[pair_evaluable]))
    pairs["absolute_log2_gain_ratio"] = gain_ratio
    pairs.to_parquet(args.output / "context_rerouting_pairs.parquet", index=False)

    correlation = spearmanr(
        pairs["module_js_divergence"], 1.0 - pairs["response_cosine"], nan_policy="omit"
    )
    audit = {
        "rows": int(len(rows)),
        "targets": int(rows["target_contrast"].nunique()),
        "genes": int(matrix.shape[1]),
        "nonzero_response_coefficients": int(matrix.nnz),
        "components": int(args.components),
        "reference_explained_variance": float(reference_model.explained_variance_ratio_.sum()),
        "median_oof_reconstruction_cosine": float(np.median(reconstruction_cosine)),
        "state_pairs": int(len(pairs)),
        "evaluable_state_pairs": int(pair_evaluable.sum()),
        "median_response_cosine": float(pairs["response_cosine"].median()),
        "median_module_js_divergence": float(pairs["module_js_divergence"].median()),
        "rerouting_metric_spearman_rho": float(correlation.statistic),
        "rerouting_metric_spearman_p": float(correlation.pvalue),
    }
    (args.output / "vector_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
