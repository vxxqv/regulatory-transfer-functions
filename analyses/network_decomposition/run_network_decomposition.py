"""Run the frozen response-network decomposition for Figure 3."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import fisher_exact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/network_decomposition/results"


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def stable_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def load_motifs(path: Path, universe: set[str], minimum: int, maximum: int) -> dict[str, set[str]]:
    motifs: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or "(human)" not in fields[0].lower():
                continue
            genes = {gene.upper() for gene in fields[2:] if gene.upper() in universe}
            if minimum <= len(genes) <= maximum:
                motifs[fields[0]] = genes
    return motifs


def motif_tests(label: str, selected: set[str], motifs: dict[str, set[str]], universe: set[str]) -> pd.DataFrame:
    records = []
    for motif, genes in motifs.items():
        overlap = len(selected & genes)
        a = overlap
        b = len(selected) - a
        c = len(genes) - a
        d = len(universe) - a - b - c
        if not selected:
            odds, p_value, ci_low, ci_high, status = np.nan, np.nan, np.nan, np.nan, "unavailable_no_response_genes"
        else:
            odds, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            cells = np.asarray([a, b, c, d], dtype=float) + 0.5
            log_odds = np.log(cells[0] * cells[3] / (cells[1] * cells[2]))
            standard_error = np.sqrt(np.sum(1.0 / cells))
            ci_low, ci_high = np.exp(log_odds - 1.96 * standard_error), np.exp(log_odds + 1.96 * standard_error)
            status = "estimable"
        records.append({"exemplar": label, "motif": motif, "selected_genes": len(selected), "motif_genes": len(genes), "overlap": a, "odds_ratio": odds, "odds_ci_low": ci_low, "odds_ci_high": ci_high, "p_value": p_value, "status": status})
    table = pd.DataFrame(records)
    table["q_value"] = np.nan
    estimable = table["status"] == "estimable"
    if estimable.any():
        table.loc[estimable, "q_value"] = bh(table.loc[estimable, "p_value"].to_numpy())
    return table.sort_values(["q_value", "p_value", "motif"], kind="stable")


def ridge_oof(table: pd.DataFrame, columns: list[str], folds: np.ndarray, outcome: np.ndarray, alpha: float) -> np.ndarray:
    prediction = np.full(len(table), np.nan)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        model = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        model.fit(table.loc[train, columns], outcome[train])
        prediction[test] = model.predict(table.loc[test, columns])
    return prediction


def bootstrap_delta(observed: np.ndarray, first: np.ndarray, second: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    _, codes = np.unique(groups, return_inverse=True)
    group_count = int(codes.max()) + 1
    values = np.empty(replicates, dtype=float)
    for _ in range(replicates):
        group_weights = rng.multinomial(group_count, np.full(group_count, 1.0 / group_count))
        weights = group_weights[codes]
        mean = np.average(observed, weights=weights)
        denominator = np.sum(weights * np.square(observed - mean))
        first_r2 = 1.0 - np.sum(weights * np.square(observed - first)) / denominator
        second_r2 = 1.0 - np.sum(weights * np.square(observed - second)) / denominator
        values[_] = first_r2 - second_r2
    return tuple(np.quantile(values, [0.025, 0.975]))


def main() -> None:
    config = yaml.safe_load((ROOT / "config/network_decomposition.yaml").read_text(encoding="utf-8"))["network_decomposition"]
    seed = int(config["seed"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vectors = ROOT / "data/interim/gwt_vectors"
    rows = pd.read_parquet(vectors / "rows.parquet").reset_index(drop=True)
    genes = pd.read_parquet(vectors / "genes.parquet").reset_index(drop=True)
    matrix = sparse.load_npz(vectors / "normalized_significant_logfc.npz").tocsr()
    transfer = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    loadings_long = pd.read_parquet(ROOT / "analyses/vectors/results/reference_module_loadings.parquet")
    tensor = pd.read_parquet(ROOT / "analyses/vectors/results/contextual_transfer_tensor.parquet")
    gain_rows = pd.read_parquet(ROOT / "analyses/vectors/results/network_gain_rows.parquet")
    if matrix.shape != (len(rows), len(genes)):
        raise ValueError("Response matrix dimensions disagree with metadata")

    condition = str(config["condition"])
    exemplar_gene = str(config["amplified_exemplar"])
    exemplar = transfer[(transfer["culture_condition"] == condition) & (transfer["target_contrast_gene_name"] == exemplar_gene)]
    if len(exemplar) != 1 or exemplar.iloc[0]["transfer_class"] != "amplified":
        raise ValueError("Frozen amplified exemplar is not uniquely available")
    exemplar = exemplar.iloc[0]
    candidates = transfer[(transfer["culture_condition"] == condition) & (transfer["transfer_class"] == "buffered")].copy()
    match_columns = list(config["matching_covariates"])
    transformed = candidates[match_columns].copy()
    exemplar_values = exemplar[match_columns].astype(float).to_numpy()
    for position, column in enumerate(match_columns):
        if column in {"cis_magnitude", "target_baseMean", "n_cells_target"}:
            transformed[column] = np.log1p(transformed[column].astype(float))
            exemplar_values[position] = np.log1p(float(exemplar[column]))
        scale = float(transformed[column].std(ddof=0))
        transformed[column] = (transformed[column] - exemplar_values[position]) / (scale if scale > 0 else 1.0)
    candidates["match_distance"] = np.sqrt(np.square(transformed.to_numpy(dtype=float)).sum(axis=1))
    candidates = candidates.sort_values(["match_distance", "target_contrast_gene_name"], kind="stable")
    comparator = candidates.iloc[0]
    candidates[["index", "target_contrast_gene_name", "culture_condition", "transfer_class", *match_columns, "match_distance"]].to_csv(OUTPUT / "buffered_match_candidates.csv", index=False)

    row_lookup = pd.Series(rows.index, index=rows["index"].astype(str))
    gene_names = genes["gene_name"].astype(str).str.upper().to_numpy()
    loading_matrix = loadings_long.pivot(index="module", columns="feature_id", values="loading").reindex(columns=genes["feature_id"].astype(str)).to_numpy()
    gene_module = np.argmax(np.abs(loading_matrix), axis=0) + 1
    network_records = []
    selected_gene_sets: dict[str, set[str]] = {}
    for label, phenotype in [("Amplified", exemplar), ("Buffered", comparator)]:
        row_number = int(row_lookup.loc[str(phenotype["index"])])
        vector = matrix.getrow(row_number).toarray().ravel()
        nonzero = np.flatnonzero(vector)
        order = nonzero[np.argsort(np.abs(vector[nonzero]))[::-1]]
        top = order[: int(config["network_top_edges"])]
        for rank, gene_index in enumerate(top, start=1):
            network_records.append({"exemplar": label, "target_gene": phenotype["target_contrast_gene_name"], "response_gene": gene_names[gene_index], "effect": vector[gene_index], "sign": "positive" if vector[gene_index] > 0 else "negative", "absolute_effect": abs(vector[gene_index]), "rank": rank, "module": int(gene_module[gene_index])})
        motif_top = order[: int(config["motif_gene_set_size"])]
        selected_gene_sets[label] = set(gene_names[motif_top])
    networks = pd.DataFrame(network_records)
    networks.to_parquet(OUTPUT / "exemplar_networks.parquet", index=False)

    universe = set(gene_names)
    motifs = load_motifs(ROOT / "work/motif/TRANSFAC_and_JASPAR_PWMs.gmt", universe, int(config["motif_minimum_measured_genes"]), int(config["motif_maximum_measured_genes"]))
    motif_table = pd.concat([motif_tests(label, selected, motifs, universe) for label, selected in selected_gene_sets.items()], ignore_index=True)
    motif_table.to_parquet(OUTPUT / "motif_enrichment_all.parquet", index=False)

    exemplar_ids = {"Amplified": str(exemplar["target_contrast"]), "Buffered": str(comparator["target_contrast"])}
    module_records = []
    for label, target in exemplar_ids.items():
        block = tensor[(tensor["target_contrast"] == target) & (tensor["condition"] == condition)].copy()
        block["exemplar"] = label
        module_records.append(block)
    modules = pd.concat(module_records, ignore_index=True)
    modules.to_parquet(OUTPUT / "module_contributions.parquet", index=False)

    sensitivity_records = []
    for label, phenotype in [("Amplified", exemplar), ("Buffered", comparator)]:
        row_number = int(row_lookup.loc[str(phenotype["index"])])
        original = matrix.getrow(row_number).toarray().ravel()
        base_scores = loading_matrix @ original
        base_energy = np.square(base_scores)
        base_energy /= base_energy.sum() if base_energy.sum() else 1.0
        block = networks[networks["exemplar"] == label]
        name_to_index = {name: idx for idx, name in enumerate(gene_names)}
        for edge in block.itertuples(index=False):
            altered = original.copy()
            altered[name_to_index[edge.response_gene]] = 0.0
            scores = loading_matrix @ altered
            energy = np.square(scores)
            energy /= energy.sum() if energy.sum() else 1.0
            cosine = float(np.dot(original, altered) / (np.linalg.norm(original) * np.linalg.norm(altered))) if np.linalg.norm(altered) else np.nan
            sensitivity_records.extend([
                {"exemplar": label, "response_gene": edge.response_gene, "rank": edge.rank, "metric": "Response norm", "change": np.linalg.norm(altered) / np.linalg.norm(original) - 1.0},
                {"exemplar": label, "response_gene": edge.response_gene, "rank": edge.rank, "metric": "Module concentration", "change": energy.max() - base_energy.max()},
                {"exemplar": label, "response_gene": edge.response_gene, "rank": edge.rank, "metric": "Module redistribution", "change": 0.5 * np.abs(energy - base_energy).sum()},
                {"exemplar": label, "response_gene": edge.response_gene, "rank": edge.rank, "metric": "Signature loss", "change": 1.0 - cosine},
            ])
    pd.DataFrame(sensitivity_records).to_parquet(OUTPUT / "edge_sensitivity.parquet", index=False)

    prediction = gain_rows.merge(
        transfer[["index", "transfer_residual", "log1p_target_baseMean", "log1p_n_cells_target", "cis_magnitude"]],
        on="index",
        how="inner",
        validate="one_to_one",
    )
    energy_wide = tensor.pivot_table(index=["target_contrast", "condition"], columns="module", values="energy_fraction").add_prefix("module_energy_").reset_index()
    prediction = prediction.merge(energy_wide, left_on=["target_contrast", "culture_condition"], right_on=["target_contrast", "condition"], how="left", validate="one_to_one")
    covariates = ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "n_guides"]
    topology = [column for column in prediction if column.startswith("module_energy_")] + ["effective_response_dimension", "sign_balance", "dominant_module_fraction", "oof_reconstruction_cosine"]
    folds = prediction["target_contrast"].astype(str).map(lambda value: stable_fold(value, int(config["prediction_folds"]), seed)).to_numpy()
    outcome = prediction["transfer_residual"].to_numpy(dtype=float)
    models = {"Covariates": covariates, "Topology": topology, "Combined": covariates + topology}
    predictions = prediction[["index", "target_contrast", "target_contrast_gene_name", "culture_condition", "transfer_residual"]].copy()
    predictions["fold"] = folds
    metric_records = []
    for model_name, columns in models.items():
        values = ridge_oof(prediction, columns, folds, outcome, float(config["prediction_alpha"]))
        predictions[model_name] = values
        metric_records.append({"model": model_name, "r2": r2_score(outcome, values), "pearson_r": np.corrcoef(outcome, values)[0, 1], "n_rows": len(values), "n_targets": prediction["target_contrast"].nunique()})
    low, high = bootstrap_delta(outcome, predictions["Combined"].to_numpy(), predictions["Covariates"].to_numpy(), prediction["target_contrast"].to_numpy(), int(config["bootstrap_replicates"]), seed + 31)
    metrics = pd.DataFrame(metric_records)
    metrics["combined_minus_covariates_ci_low"] = low
    metrics["combined_minus_covariates_ci_high"] = high
    predictions.to_parquet(OUTPUT / "residual_gain_predictions.parquet", index=False)
    metrics.to_csv(OUTPUT / "residual_gain_metrics.csv", index=False)

    null_path = ROOT / str(config["external_degree_null"])
    null_summary_path = ROOT / str(config["external_degree_summary"])
    degree_null = pd.read_parquet(null_path)
    degree_summary = pd.read_csv(null_summary_path)
    degree_null.to_parquet(OUTPUT / "external_degree_preserving_null.parquet", index=False)
    degree_row = degree_summary[degree_summary["null"] == "directed_degree_preserving_rewiring"]
    degree_row.to_csv(OUTPUT / "external_degree_preserving_summary.csv", index=False)

    audit = {
        "amplified_exemplar": exemplar_gene,
        "buffered_comparator": str(comparator["target_contrast_gene_name"]),
        "buffered_candidate_denominator": int(len(candidates)),
        "match_covariates": match_columns,
        "motif_library_human_eligible": int(len(motifs)),
        "motif_tests_total": int(len(motif_table)),
        "motif_tests_estimable": int((motif_table["status"] == "estimable").sum()),
        "motif_tests_unavailable": int((motif_table["status"] != "estimable").sum()),
        "motif_fdr_significant": int((motif_table["q_value"] <= float(config["multiplicity_alpha"])).sum()),
        "prediction_rows": int(len(prediction)),
        "prediction_targets": int(prediction["target_contrast"].nunique()),
        "combined_minus_covariates_r2": float(metrics.set_index("model").loc["Combined", "r2"] - metrics.set_index("model").loc["Covariates", "r2"]),
        "combined_minus_covariates_ci": [float(low), float(high)],
        "degree_null_reused_not_recomputed": True,
        "degree_null_replicates": int(len(degree_null)),
    }
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
