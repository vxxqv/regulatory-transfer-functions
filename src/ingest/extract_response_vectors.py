"""Extract signed perturbation-response vectors from the public backed H5AD object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/interim/gwt_vectors"))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--fdr", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.models.empirical_transfer import quality_filter

    args.output.mkdir(parents=True, exist_ok=True)
    source_summary = pd.read_csv(args.summary)
    selected = source_summary.loc[quality_filter(source_summary)].copy()
    selected_key = selected["index"].astype(str)

    data = ad.read_h5ad(args.h5ad, backed="r")
    row_index = data.obs_names.get_indexer(selected_key)
    if (row_index < 0).any():
        missing = selected_key[row_index < 0].head(10).tolist()
        raise ValueError(f"Selected summary rows missing from H5AD: {missing}")

    order = np.argsort(row_index)
    row_index = row_index[order]
    selected = selected.iloc[order].reset_index(drop=True)
    var_names = pd.Index(data.var_names.astype(str))
    gene_name = (
        data.var["gene_name"].astype(str)
        if "gene_name" in data.var.columns
        else pd.Series(var_names, index=var_names)
    )
    gene_table = data.var.copy()
    gene_table.insert(0, "feature_id", var_names)
    if "gene_name" not in gene_table.columns:
        gene_table["gene_name"] = gene_name.to_numpy()
    gene_table.reset_index(drop=True).to_parquet(args.output / "genes.parquet", index=False)

    cis_columns = var_names.get_indexer(selected["target_contrast"].astype(str))
    if (cis_columns < 0).any():
        missing = selected.loc[cis_columns < 0, "target_contrast"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Perturbed Ensembl IDs missing from measured genes: {missing}")

    normalized_blocks: list[sparse.csr_matrix] = []
    metrics: list[pd.DataFrame] = []
    for start in range(0, len(selected), args.chunk_size):
        stop = min(start + args.chunk_size, len(selected))
        source_rows = row_index[start:stop]
        log_fc = np.asarray(data.layers["log_fc"][source_rows, :], dtype=np.float64)
        lfc_se = np.asarray(data.layers["lfcSE"][source_rows, :], dtype=np.float64)
        adjusted = np.asarray(data.layers["adj_p_value"][source_rows, :], dtype=np.float64)
        cis = cis_columns[start:stop]
        local_rows = np.arange(stop - start)

        log_fc[local_rows, cis] = 0.0
        lfc_se[local_rows, cis] = 0.0
        adjusted[local_rows, cis] = np.nan
        significant = np.isfinite(adjusted) & (adjusted <= args.fdr)
        finite_effect = np.where(np.isfinite(log_fc), log_fc, 0.0)
        finite_se = np.where(np.isfinite(lfc_se), lfc_se, 0.0)
        cis_magnitude = selected.loc[start : stop - 1, "ontarget_effect_size"].abs().to_numpy()
        scale = np.maximum(cis_magnitude, 0.1)

        normalized = np.where(significant, finite_effect / scale[:, None], 0.0)
        normalized_blocks.append(sparse.csr_matrix(normalized.astype(np.float32)))
        variance_corrected = np.maximum(finite_effect**2 - finite_se**2, 0.0)
        significant_effect = np.where(significant, finite_effect, 0.0)
        positive = significant & (finite_effect > 0)
        negative = significant & (finite_effect < 0)
        block = pd.DataFrame(
            {
                "trans_l2_naive": np.sqrt(np.sum(finite_effect**2, axis=1)),
                "trans_l2_debiased": np.sqrt(np.sum(variance_corrected, axis=1)),
                "trans_l2_significant": np.sqrt(np.sum(significant_effect**2, axis=1)),
                "trans_l1_significant": np.sum(np.abs(significant_effect), axis=1),
                "positive_significant": np.sum(positive, axis=1),
                "negative_significant": np.sum(negative, axis=1),
                "significant_vector_size": np.sum(significant, axis=1),
                "signed_sum_significant": np.sum(significant_effect, axis=1),
            }
        )
        block["gamma_l2_debiased"] = block["trans_l2_debiased"] / scale
        block["gamma_l2_significant"] = block["trans_l2_significant"] / scale
        block["gamma_l1_significant"] = block["trans_l1_significant"] / scale
        block["effective_response_dimension"] = np.where(
            block["trans_l2_significant"] > 0,
            (block["trans_l1_significant"] ** 2)
            / np.maximum(block["trans_l2_significant"] ** 2, np.finfo(float).eps),
            0.0,
        )
        block["sign_balance"] = (
            (block["positive_significant"] - block["negative_significant"])
            / np.maximum(block["significant_vector_size"], 1)
        )
        metrics.append(block)

    response = sparse.vstack(normalized_blocks, format="csr")
    sparse.save_npz(args.output / "normalized_significant_logfc.npz", response, compressed=True)
    vector_metrics = pd.concat(metrics, ignore_index=True)
    rows = pd.concat([selected, vector_metrics], axis=1)
    rows["significant_count_matches_source"] = (
        rows["significant_vector_size"].astype(int) == rows["n_downstream"].astype(int)
    )
    rows.to_parquet(args.output / "rows.parquet", index=False)

    audit = {
        "h5ad_rows": int(data.n_obs),
        "h5ad_genes": int(data.n_vars),
        "selected_rows": int(len(rows)),
        "selected_targets": int(rows["target_contrast"].nunique()),
        "matrix_nonzero": int(response.nnz),
        "matrix_density": float(response.nnz / (response.shape[0] * response.shape[1])),
        "significant_count_match_fraction": float(rows["significant_count_matches_source"].mean()),
        "fdr": float(args.fdr),
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data.file.close()


if __name__ == "__main__":
    main()
