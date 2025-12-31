"""Validate perturbational transfer directions using independent eQTLGen effects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import binomtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=Path("data/interim/gwt_vectors"))
    parser.add_argument("--eqtlgen", type=Path, default=Path("data/interim/eqtlgen"))
    parser.add_argument("--output", type=Path, default=Path("analyses/natural_genetics/results"))
    parser.add_argument("--config", type=Path, default=Path("config/analysis.yaml"))
    parser.add_argument("--permutations", type=int, default=1000)
    return parser.parse_args()


def cluster_bootstrap_interval(
    table: pd.DataFrame, cluster: str, outcome: str, replicates: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = [group[outcome].to_numpy() for _, group in table.groupby(cluster, sort=False)]
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draws = rng.integers(0, len(grouped), len(grouped))
        sample = np.concatenate([grouped[position] for position in draws])
        estimates[index] = np.mean(sample)
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.models.natural_genetics import allele_orientation

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["study"]["seed"])
    args.output.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(args.vectors / "rows.parquet").reset_index(drop=True)
    genes = pd.read_parquet(args.vectors / "genes.parquet")
    matrix = sparse.load_npz(args.vectors / "normalized_significant_logfc.npz").tocsr()
    cis = pd.read_parquet(args.eqtlgen / "cis_mediators.parquet")
    trans = pd.read_parquet(args.eqtlgen / "trans_associations.parquet")
    gene_lookup = pd.Series(np.arange(len(genes)), index=genes["feature_id"].astype(str)).to_dict()
    response_lookup = rows.reset_index(names="response_row").set_index(
        ["target_contrast", "culture_condition"]
    )["response_row"]

    tests = cis.merge(trans, on="SNP", suffixes=("_cis", "_trans"), validate="one_to_many")
    tests["gene_column"] = tests["Gene_trans"].astype(str).map(gene_lookup)
    tests = tests.dropna(subset=["gene_column"]).copy()
    tests["gene_column"] = tests["gene_column"].astype(int)
    records: list[pd.DataFrame] = []
    for condition in config["primary_resource"]["conditions"]:
        condition_tests = tests.copy()
        keys = pd.MultiIndex.from_arrays(
            [condition_tests["Gene_cis"].astype(str), np.repeat(condition, len(condition_tests))]
        )
        condition_tests["response_row"] = response_lookup.reindex(keys).to_numpy()
        condition_tests = condition_tests.dropna(subset=["response_row"]).copy()
        condition_tests["response_row"] = condition_tests["response_row"].astype(int)
        orientations = np.array(
            [
                allele_orientation(a, b, c, d)
                for a, b, c, d in zip(
                    condition_tests["AssessedAllele_cis"],
                    condition_tests["OtherAllele_cis"],
                    condition_tests["AssessedAllele_trans"],
                    condition_tests["OtherAllele_trans"],
                    strict=True,
                )
            ]
        )
        condition_tests["allele_orientation"] = orientations
        condition_tests = condition_tests.loc[condition_tests["allele_orientation"] != 0].copy()
        extracted = matrix[
            condition_tests["response_row"].to_numpy(), condition_tests["gene_column"].to_numpy()
        ]
        condition_tests["perturbation_effect"] = np.asarray(extracted).ravel()
        condition_tests = condition_tests.loc[condition_tests["perturbation_effect"] != 0].copy()
        aligned_trans_z = (
            condition_tests["Zscore_trans"].to_numpy()
            * condition_tests["allele_orientation"].to_numpy()
        )
        condition_tests["predicted_direction"] = -np.sign(
            condition_tests["Zscore_cis"].to_numpy()
        ) * np.sign(condition_tests["perturbation_effect"].to_numpy())
        condition_tests["observed_direction"] = np.sign(aligned_trans_z)
        condition_tests["direction_match"] = (
            condition_tests["predicted_direction"] == condition_tests["observed_direction"]
        ).astype(int)
        condition_tests["condition"] = condition
        records.append(condition_tests)
    pairs = pd.concat(records, ignore_index=True)
    if pairs.empty:
        raise ValueError("No significant perturbational responses overlap eQTLGen trans associations")

    response_degree = np.asarray((matrix != 0).sum(axis=1)).ravel()
    rows["degree_bin"] = rows.groupby("culture_condition")["target_contrast"].transform(
        lambda _: 0
    )
    for condition, index in rows.groupby("culture_condition").groups.items():
        values = response_degree[np.asarray(list(index), dtype=int)]
        ranks = pd.Series(values).rank(method="first")
        rows.loc[list(index), "degree_bin"] = pd.qcut(ranks, 10, labels=False, duplicates="drop").to_numpy()
    pairs["degree_bin"] = rows.loc[pairs["response_row"], "degree_bin"].to_numpy(dtype=int)

    rng = np.random.default_rng(seed)
    null = np.empty(args.permutations, dtype=float)
    candidate_rows = {
        key: group.index.to_numpy(dtype=int)
        for key, group in rows.groupby(["culture_condition", "degree_bin"])
    }
    observed_direction = pairs["observed_direction"].to_numpy()
    gene_columns = pairs["gene_column"].to_numpy(dtype=int)
    cis_sign = np.sign(pairs["Zscore_cis"].to_numpy())
    strata = list(zip(pairs["condition"], pairs["degree_bin"], strict=True))
    for permutation in range(args.permutations):
        randomized_rows = np.array(
            [rng.choice(candidate_rows[stratum]) for stratum in strata], dtype=int
        )
        randomized_effect = np.asarray(matrix[randomized_rows, gene_columns]).ravel()
        randomized_prediction = -cis_sign * np.sign(randomized_effect)
        evaluable = randomized_effect != 0
        null[permutation] = (
            np.mean(randomized_prediction[evaluable] == observed_direction[evaluable])
            if evaluable.any()
            else np.nan
        )

    agreement = float(pairs["direction_match"].mean())
    ci_low, ci_high = cluster_bootstrap_interval(
        pairs, "SNP", "direction_match", replicates=2000, seed=seed + 1
    )
    condition_summary = (
        pairs.groupby("condition", as_index=False)
        .agg(
            pairs=("direction_match", "size"),
            snps=("SNP", "nunique"),
            mediators=("Gene_cis", "nunique"),
            direction_agreement=("direction_match", "mean"),
        )
        .sort_values("condition")
    )
    chromosome_holdout = (
        pairs.groupby("SNPChr_cis", as_index=False)
        .agg(
            pairs=("direction_match", "size"),
            snps=("SNP", "nunique"),
            direction_agreement=("direction_match", "mean"),
        )
        .sort_values("SNPChr_cis")
    )
    sensitivity = pairs.loc[pairs["BonferroniP_trans"] <= 0.05]
    exact = binomtest(int(pairs["direction_match"].sum()), len(pairs), 0.5, alternative="greater")
    metrics = {
        "pairs": int(len(pairs)),
        "snps": int(pairs["SNP"].nunique()),
        "mediators": int(pairs["Gene_cis"].nunique()),
        "trans_genes": int(pairs["Gene_trans"].nunique()),
        "direction_agreement": agreement,
        "snp_cluster_bootstrap_ci_95": [ci_low, ci_high],
        "binomial_p_descriptive": float(exact.pvalue),
        "matched_permutation_null_median": float(np.nanmedian(null)),
        "matched_permutation_p_upper": float(
            (1 + np.sum(null >= agreement)) / (1 + np.sum(np.isfinite(null)))
        ),
        "bonferroni_trans_pairs": int(len(sensitivity)),
        "bonferroni_trans_direction_agreement": (
            float(sensitivity["direction_match"].mean()) if len(sensitivity) else None
        ),
    }
    pairs.to_parquet(args.output / "directional_pairs.parquet", index=False)
    pd.DataFrame({"direction_agreement": null}).to_parquet(
        args.output / "matched_null.parquet", index=False
    )
    condition_summary.to_csv(args.output / "condition_summary.csv", index=False)
    chromosome_holdout.to_csv(args.output / "chromosome_holdout.csv", index=False)
    (args.output / "natural_genetics_results.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
