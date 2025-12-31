"""Build frozen context-dynamics tables for Figure 5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/context_dynamics/results"
sys.path.insert(0, str(ROOT))
from analyses.network_decomposition.run_network_decomposition import load_motifs


def target_bootstrap(x: np.ndarray, y: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for replicate in range(replicates):
        chosen = rng.integers(0, len(x), len(x))
        values[replicate] = spearmanr(x[chosen], y[chosen]).statistic
    return tuple(np.nanquantile(values, [0.025, 0.975]))


def finite_range(values: pd.Series) -> float:
    values = values.dropna().to_numpy(dtype=float)
    return float(values.max() - values.min()) if len(values) else np.nan


def main() -> None:
    config = yaml.safe_load((ROOT / "config/context_dynamics.yaml").read_text(encoding="utf-8"))["context_dynamics"]
    states = list(config["states"])
    seed = int(config["seed"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    transfer = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    switching = pd.read_parquet(ROOT / "analyses/primary/results/context_switching.parquet")
    tensor = pd.read_parquet(ROOT / "analyses/vectors/results/contextual_transfer_tensor.parquet")
    manifold = pd.read_parquet(ROOT / "analyses/manifold/results/response_manifold.parquet")
    rows = pd.read_parquet(ROOT / "data/interim/gwt_vectors/rows.parquet").reset_index(drop=True)
    genes = pd.read_parquet(ROOT / "data/interim/gwt_vectors/genes.parquet").reset_index(drop=True)
    matrix = sparse.load_npz(ROOT / "data/interim/gwt_vectors/normalized_significant_logfc.npz").tocsr()

    complete = transfer.groupby("target_contrast")["culture_condition"].nunique()
    complete_targets = complete[complete == len(states)].index
    switching = switching[switching["target_contrast"].isin(complete_targets)].sort_values(["residual_range", "target_contrast"], ascending=[False, True], kind="stable")
    selected = switching.head(int(config["selected_targets"]))[["target_contrast", "target_contrast_gene_name", "residual_range", "cluster_entropy", "cluster_switch"]]
    selected.to_csv(OUTPUT / "selected_switch_targets.csv", index=False)
    manifold_selected = manifold[manifold["target_contrast"].isin(selected["target_contrast"])].merge(
        selected.drop(columns="target_contrast_gene_name"),
        on="target_contrast",
        how="left",
        validate="many_to_one",
    )
    manifold_selected.to_parquet(OUTPUT / "selected_manifold.parquet", index=False)

    selected_tensor = tensor[tensor["target_contrast"].isin(selected["target_contrast"])].copy()
    variable_modules = selected_tensor.groupby("module")["energy_fraction"].var().nlargest(int(config["tensor_modules"])).index
    selected_tensor = selected_tensor[selected_tensor["module"].isin(variable_modules)]
    selected_tensor.to_parquet(OUTPUT / "selected_context_tensor.parquet", index=False)

    stable = transfer[transfer["target_contrast"].isin(complete_targets)].groupby(["target_contrast", "target_contrast_gene_name"]).agg(cis_mean=("cis_magnitude", "mean"), cis_sd=("cis_magnitude", "std")).reset_index()
    stable["cis_cv"] = stable["cis_sd"] / stable["cis_mean"]
    stable = stable.merge(switching[["target_contrast", "residual_range"]], on="target_contrast", how="inner", validate="one_to_one")
    stable = stable[stable["cis_cv"] <= float(config["stable_cis_cv_maximum"])].sort_values(["residual_range", "target_contrast"], ascending=[False, True], kind="stable")
    examples = stable.head(int(config["stable_cis_examples"]))
    example_states = transfer[transfer["target_contrast"].isin(examples["target_contrast"])].merge(examples[["target_contrast", "cis_cv", "residual_range"]], on="target_contrast", how="left", validate="many_to_one")
    example_states.to_parquet(OUTPUT / "stable_cis_variable_trans.parquet", index=False)

    energy_by_state = tensor[tensor["target_contrast"].isin(selected["target_contrast"])].groupby(["target_contrast", "condition"])["energy_fraction"].sum().unstack()
    all_states_nonzero = energy_by_state.index[(energy_by_state.reindex(columns=states) > 0).all(axis=1)]
    alluvial_candidates = selected[selected["cluster_switch"] & selected["target_contrast"].isin(all_states_nonzero)]
    alluvial_target = alluvial_candidates.iloc[0] if len(alluvial_candidates) else selected.iloc[0]
    alluvial = tensor[tensor["target_contrast"] == alluvial_target["target_contrast"]].copy()
    alluvial["is_top_module"] = alluvial["module"].isin(alluvial.groupby("module")["energy_fraction"].max().nlargest(6).index)
    alluvial.loc[~alluvial["is_top_module"], "module"] = 0
    alluvial = alluvial.groupby(["target_contrast", "target_gene", "condition", "module"], as_index=False)["energy_fraction"].sum()
    alluvial.to_parquet(OUTPUT / "alluvial_programs.parquet", index=False)

    null = pd.read_parquet(ROOT / "analyses/primary/results/context_switch_null.parquet")
    switching.to_parquet(OUTPUT / "switch_scores.parquet", index=False)
    null.to_parquet(OUTPUT / "switch_null.parquet", index=False)

    universe = set(genes["gene_name"].astype(str).str.upper())
    motifs = load_motifs(ROOT / "work/motif/TRANSFAC_and_JASPAR_PWMs.gmt", universe, int(config["tf_activity_minimum_measured_genes"]), int(config["tf_activity_maximum_measured_genes"]))
    name_to_index = pd.Series(genes.index, index=genes["gene_name"].astype(str).str.upper())
    membership = sparse.lil_matrix((len(genes), len(motifs)), dtype=float)
    motif_names = list(motifs)
    for motif_index, motif_name in enumerate(motif_names):
        indices = name_to_index.reindex(sorted(motifs[motif_name])).dropna().astype(int).to_numpy()
        membership[indices, motif_index] = 1.0 / len(indices)
    activity = (matrix @ membership.tocsr()).toarray()
    activity_rows = pd.DataFrame({"target_contrast": rows["target_contrast"].astype(str), "target_gene": rows["target_contrast_gene_name"].astype(str), "condition": rows["culture_condition"].astype(str)})
    activity_summary = []
    row_groups = activity_rows.groupby("target_contrast").indices
    switch_lookup = switching.set_index("target_contrast")["residual_range"]
    for target in complete_targets:
        if target not in row_groups or target not in switch_lookup:
            continue
        indices = np.asarray(row_groups[target])
        if len(indices) != len(states):
            continue
        ranges = np.ptp(activity[indices], axis=0)
        best = int(np.argmax(ranges))
        activity_summary.append({"target_contrast": target, "target_gene": activity_rows.iloc[indices[0]]["target_gene"], "transfer_switch_score": switch_lookup.loc[target], "maximum_tf_activity_range": ranges[best], "top_tf_motif": motif_names[best]})
    activity_summary = pd.DataFrame(activity_summary)
    gain_rows = pd.read_parquet(ROOT / "analyses/vectors/results/network_gain_rows.parquet")
    global_summary = gain_rows[gain_rows["target_contrast"].isin(complete_targets)].groupby("target_contrast").agg(
        mean_response_norm=("response_norm", "mean"),
        range_response_norm=("response_norm", lambda value: float(value.max() - value.min())),
        mean_n_downstream=("n_downstream", "mean"),
        mean_cis_magnitude=("ontarget_effect_size", lambda value: float(np.mean(np.abs(value)))),
        mean_target_expression=("target_baseMean", "mean"),
        mean_target_cells=("n_cells_target", "mean"),
        mean_guides=("n_guides", "mean"),
    ).reset_index()
    activity_summary = activity_summary.merge(global_summary, on="target_contrast", how="left", validate="one_to_one")
    adjustment_columns = ["mean_response_norm", "range_response_norm", "mean_n_downstream", "mean_cis_magnitude", "mean_target_expression", "mean_target_cells", "mean_guides"]
    adjustment = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    activity_summary["adjusted_tf_activity_range"] = activity_summary["maximum_tf_activity_range"] - adjustment.fit(activity_summary[adjustment_columns], activity_summary["maximum_tf_activity_range"]).predict(activity_summary[adjustment_columns])
    switch_adjustment = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    activity_summary["adjusted_transfer_switch_score"] = activity_summary["transfer_switch_score"] - switch_adjustment.fit(activity_summary[adjustment_columns], activity_summary["transfer_switch_score"]).predict(activity_summary[adjustment_columns])
    tf_corr = spearmanr(activity_summary["adjusted_transfer_switch_score"], activity_summary["adjusted_tf_activity_range"])
    tf_low, tf_high = target_bootstrap(activity_summary["adjusted_transfer_switch_score"].to_numpy(), activity_summary["adjusted_tf_activity_range"].to_numpy(), int(config["bootstrap_replicates"]), seed + 41)
    activity_summary.to_parquet(OUTPUT / "tf_activity_association.parquet", index=False)

    validation = pd.read_csv(ROOT / "work/upstream/GWT_perturbseq_analysis_2025/metadata/suppl_tables/Th1Th2_validation_summary.suppl_table.csv")
    external = validation.groupby("target_name").agg(external_state_range=("bulkRNA_Th1_mean_zscore", finite_range), external_th2_range=("bulkRNA_Th2_mean_zscore", finite_range), external_states=("condition", "nunique"), external_donors=("bulkRNA_n_donors", "max")).reset_index()
    external["external_combined_range"] = np.sqrt(np.square(external["external_state_range"]) + np.square(external["external_th2_range"]))
    validation_table = switching[["target_contrast_gene_name", "residual_range"]].merge(external, left_on="target_contrast_gene_name", right_on="target_name", how="inner")
    validation_table = validation_table[validation_table["external_states"] >= 2].copy()
    external_corr = spearmanr(validation_table["residual_range"], validation_table["external_combined_range"])
    if len(validation_table) >= 4:
        ext_low, ext_high = target_bootstrap(validation_table["residual_range"].to_numpy(), validation_table["external_combined_range"].to_numpy(), int(config["bootstrap_replicates"]), seed + 73)
    else:
        ext_low, ext_high = np.nan, np.nan
    validation_table.to_csv(OUTPUT / "independent_state_validation.csv", index=False)

    audit = {
        "complete_three_state_targets": int(len(complete_targets)),
        "selected_switch_targets": int(len(selected)),
        "stable_cis_eligible_targets": int(len(stable)),
        "alluvial_target": str(alluvial_target["target_contrast_gene_name"]),
        "tf_motif_denominator": int(len(motifs)),
        "tf_activity_targets": int(len(activity_summary)),
        "tf_activity_spearman_rho": float(tf_corr.statistic),
        "tf_activity_spearman_p": float(tf_corr.pvalue),
        "tf_activity_bootstrap_ci": [float(tf_low), float(tf_high)],
        "independent_state_targets": int(len(validation_table)),
        "independent_state_spearman_rho": float(external_corr.statistic),
        "independent_state_spearman_p": float(external_corr.pvalue),
        "independent_state_bootstrap_ci": [float(ext_low), float(ext_high)],
        "independent_state_underpowered": bool(len(validation_table) < 20 or ext_low <= 0 <= ext_high),
    }
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
