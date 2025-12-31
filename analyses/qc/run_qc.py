"""Create the immutable primary quality-control report and Supplementary Figures S1-S4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("analyses/qc/results"))
    parser.add_argument("--figure-root", type=Path, default=Path("figures/supplement"))
    parser.add_argument("--config", type=Path, default=Path("config/analysis.yaml"))
    parser.add_argument("--colors", type=Path, default=Path("config/colors.yaml"))
    return parser.parse_args()


def label_grid(figure: plt.Figure) -> None:
    for label, x, y in (("A", 0.01, 0.99), ("B", 0.51, 0.99), ("C", 0.01, 0.50), ("D", 0.51, 0.50)):
        figure.text(x, y, label, fontsize=14, fontweight="bold", va="top")


def save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / f"{destination.name}.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(destination / f"{destination.name}.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(destination / f"{destination.name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.models.empirical_transfer import quality_filter
    from src.qc.summary_qc import audit_counts

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    colors = yaml.safe_load(args.colors.read_text(encoding="utf-8"))["states"]
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    source = args.source_root / "metadata" / "suppl_tables"
    args.output.mkdir(parents=True, exist_ok=True)
    lanes = pd.read_csv(source / "QC_summaries_per_sample_lane.csv")
    samples = pd.read_csv(source / "sample_metadata.suppl_table.csv")
    guides = pd.read_csv(source / "sgrna_library_metadata.suppl_table.csv")
    de = pd.read_csv(source / "DE_stats.suppl_table.csv")
    qc_config = config["quality_control"]
    selected = de.loc[
        quality_filter(de, qc_config["minimum_guides"], qc_config["minimum_cells"])
    ].copy()
    cascade = audit_counts(de, qc_config["minimum_guides"], qc_config["minimum_cells"])

    lanes["culture_condition"] = lanes["library_id"].str.extract(r"_(Rest|Stim8hr|Stim48hr)_")
    assignment_columns = [
        "NTC single sgRNA",
        "multi sgRNA",
        "no sgRNA (>= 3 UMIs)",
        "targeting single sgRNA",
    ]
    assignment = lanes[assignment_columns].sum().rename_axis("assignment").reset_index(name="cells")
    condition_rows = (
        selected.groupby("culture_condition", as_index=False)
        .agg(rows=("index", "size"), targets=("target_contrast", "nunique"))
        .sort_values("culture_condition")
    )
    lanes.to_parquet(args.output / "lane_qc.parquet", index=False)
    guides.to_parquet(args.output / "guide_qc.parquet", index=False)
    cascade.to_csv(args.output / "filter_cascade.csv", index=False)
    assignment.to_csv(args.output / "guide_assignment.csv", index=False)
    condition_rows.to_csv(args.output / "condition_rows.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    axes[0].hist(lanes["n_cells"], bins=24, color="#4C78A8", edgecolor="white")
    axes[0].set(xlabel="Cells per lane", ylabel="Lanes")
    axes[1].scatter(lanes["mean_total_counts"], lanes["mean_n_genes"], s=11, alpha=0.7, color="#2F4B7C")
    axes[1].set(xlabel="Mean molecules per cell", ylabel="Mean genes per cell")
    axes[2].hist(lanes["mean_pct_counts_mt"], bins=24, color="#59A14F", edgecolor="white")
    axes[2].set(xlabel="Mean mitochondrial fraction (%)", ylabel="Lanes")
    axes[3].barh(assignment["assignment"], assignment["cells"] / 1e6, color=["#76B7B2", "#E15759", "#BAB0AC", "#F28E2B"])
    axes[3].set(xlabel="Cells (millions)")
    label_grid(fig)
    save_figure(fig, args.figure_root / "S01")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    guide_counts = guides.groupby("target_gene_id")["sgRNA"].nunique()
    axes[0].hist(guide_counts, bins=np.arange(0.5, guide_counts.max() + 1.5), color="#4C78A8", edgecolor="white")
    axes[0].set(xlabel="Guides per designed target", ylabel="Targets")
    positive_distance = guides["distance_to_closest_target_tss"].dropna().abs() + 1
    axes[1].hist(np.log10(positive_distance), bins=30, color="#F28E2B", edgecolor="white")
    axes[1].set(xlabel="log10 distance to target TSS + 1", ylabel="Guides")
    flags = pd.Series(
        {
            "Flagged": guides["flag"].astype("string").str.lower().eq("true").fillna(False).sum(),
            "Bidirectional": guides["putative_bidirectional_promoter"].astype("string").str.lower().eq("true").fillna(False).sum(),
            "Alternate alignment": guides["other_alignment_chromosome"].notna().sum(),
        }
    )
    axes[2].bar(flags.index, flags.values, color=["#E15759", "#B07AA1", "#9C755F"])
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set(ylabel="Guides")
    hb = axes[3].hexbin(selected["n_cells_target"], selected["n_guides"], gridsize=35, mincnt=1, bins="log", cmap="viridis")
    axes[3].set(xlabel="Cells per target-state test", ylabel="Guides")
    fig.colorbar(hb, ax=axes[3], label="log10 tests")
    label_grid(fig)
    save_figure(fig, args.figure_root / "S02")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    guide_corr = selected["guide_correlation_all"].dropna()
    donor_corr = selected["donor_correlation_all_mean"].dropna()
    axes[0].hist(guide_corr, bins=35, color="#4C78A8", edgecolor="white")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set(xlabel="Between-guide correlation", ylabel="Tests")
    axes[1].hist(donor_corr, bins=35, color="#59A14F", edgecolor="white")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Mean between-donor correlation", ylabel="Tests")
    complete = selected[["guide_correlation_all", "donor_correlation_all_mean"]].dropna()
    axes[2].hexbin(complete.iloc[:, 0], complete.iloc[:, 1], gridsize=28, mincnt=1, bins="log", cmap="magma")
    axes[2].set(xlabel="Guide correlation", ylabel="Donor correlation")
    state_data = [selected.loc[selected["culture_condition"] == state, "guide_correlation_all"].dropna() for state in colors]
    violin = axes[3].violinplot(state_data, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], colors.values(), strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.8)
    display_states = {"Rest": "Rest", "Stim8hr": "8 h", "Stim48hr": "48 h"}
    axes[3].set_xticks(
        range(1, len(colors) + 1), [display_states[state] for state in colors], rotation=0
    )
    axes[3].set(ylabel="Guide correlation")
    label_grid(fig)
    save_figure(fig, args.figure_root / "S03")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    axes[0].barh(cascade["criterion"], cascade["fraction_rows"], color="#4C78A8")
    axes[0].set(xlabel="Fraction of source tests", xlim=(0, 1.02))
    axes[1].hist(selected["ontarget_effect_size"], bins=45, color="#F28E2B", edgecolor="white")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="On-target log2 fold change", ylabel="Tests")
    hb = axes[2].hexbin(np.abs(selected["ontarget_effect_size"]), np.log1p(selected["n_downstream"]), gridsize=40, mincnt=1, bins="log", cmap="viridis")
    axes[2].set(xlabel="Absolute on-target effect", ylabel="log1p downstream genes")
    fig.colorbar(hb, ax=axes[2], label="log10 tests")
    axes[3].bar(
        [display_states[state] for state in condition_rows["culture_condition"]],
        condition_rows["targets"],
        color=[colors[state] for state in condition_rows["culture_condition"]],
    )
    axes[3].tick_params(axis="x", rotation=0)
    axes[3].set(ylabel="Primary targets")
    label_grid(fig)
    save_figure(fig, args.figure_root / "S04")

    report = {
        "lanes": int(len(lanes)),
        "raw_lane_cell_sum": int(lanes["n_cells"].sum()),
        "lane_low_quality_cell_sum": int(lanes["n_low_quality_cells"].sum()),
        "samples": int(len(samples)),
        "donors": int(samples["donor_id"].nunique()),
        "conditions": int(samples["culture_condition"].nunique()),
        "guides": int(guides["sgRNA"].nunique()),
        "designed_targets": int(guides["target_gene_id"].nunique()),
        "source_target_state_tests": int(len(de)),
        "primary_target_state_tests": int(len(selected)),
        "primary_targets": int(selected["target_contrast"].nunique()),
    }
    (args.output / "qc_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
