"""Create the immutable primary quality-control report and Supplementary Figures S1-S4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
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


def label_grid(axes: np.ndarray) -> None:
    """Place panel labels relative to each axis so tight bounding boxes cannot drop them."""
    for label, axis in zip("ABCD", axes, strict=True):
        axis.text(
            -0.14,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )


def save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / f"{destination.name}.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(destination / f"{destination.name}.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(destination / f"{destination.name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_figure_data(destination: Path, name: str, data: pd.DataFrame) -> None:
    figure_data = destination / "figure_data"
    figure_data.mkdir(parents=True, exist_ok=True)
    data.to_csv(figure_data / name, sep="\t", index=False)


def apply_density_scale(collection: mpl.collections.Collection, name: str) -> tuple[list[int], list[int]]:
    """Apply an adaptive discrete log scale that remains vector in SVG and PDF."""
    candidates = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
    maximum = max(1, int(np.nanmax(collection.get_array())))
    upper_index = next(index for index, value in enumerate(candidates) if value > maximum)
    boundaries = candidates[: upper_index + 1]
    base = mpl.colormaps[name]
    colors = base(np.linspace(0.08, 0.92, len(boundaries) - 1))
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    collection.set_cmap(cmap)
    collection.set_norm(norm)
    ticks = [value for value in [1, 10, 100, 1000, 10000] if value <= boundaries[-1]]
    return boundaries, ticks


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
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
    axes[0].set_title("Lane yield")
    axes[0].set(xlabel="Cells per lane", ylabel="Lanes")
    axes[1].scatter(lanes["mean_total_counts"], lanes["mean_n_genes"], s=11, alpha=0.7, color="#2F4B7C")
    axes[1].set_title("Library complexity")
    axes[1].set(xlabel="Mean molecules per cell", ylabel="Mean genes per cell")
    axes[2].hist(lanes["mean_pct_counts_mt"], bins=24, color="#59A14F", edgecolor="white")
    axes[2].set_title("Mitochondrial fraction")
    axes[2].set(xlabel="Mean mitochondrial fraction (%)", ylabel="Lanes")
    axes[3].barh(assignment["assignment"], assignment["cells"] / 1e6, color=["#76B7B2", "#E15759", "#BAB0AC", "#F28E2B"])
    axes[3].set_title("Guide assignment")
    axes[3].set(xlabel="Cells (millions)")
    label_grid(axes)
    destination = args.figure_root / "S01"
    write_figure_data(destination, "A_lane_cells.tsv", lanes[["library_id", "n_cells"]])
    write_figure_data(destination, "B_lane_complexity.tsv", lanes[["library_id", "mean_total_counts", "mean_n_genes"]])
    write_figure_data(destination, "C_lane_mitochondrial_fraction.tsv", lanes[["library_id", "mean_pct_counts_mt"]])
    write_figure_data(destination, "D_guide_assignment.tsv", assignment)
    save_figure(fig, destination)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    guide_counts = guides.groupby("target_gene_id")["sgRNA"].nunique()
    axes[0].hist(guide_counts, bins=np.arange(0.5, guide_counts.max() + 1.5), color="#4C78A8", edgecolor="white")
    axes[0].set_title("Target coverage")
    axes[0].set(xlabel="Guides per designed target", ylabel="Targets")
    positive_distance = guides["distance_to_closest_target_tss"].dropna().abs() + 1
    axes[1].hist(np.log10(positive_distance), bins=30, color="#F28E2B", edgecolor="white")
    axes[1].set_title("TSS distance")
    axes[1].set(xlabel="log10 distance to target TSS + 1", ylabel="Guides")
    flags = pd.Series(
        {
            "Flagged": guides["flag"].astype("string").str.lower().eq("true").fillna(False).sum(),
            "Bidirectional": guides["putative_bidirectional_promoter"].astype("string").str.lower().eq("true").fillna(False).sum(),
            "Alternate alignment": guides["other_alignment_chromosome"].notna().sum(),
        }
    )
    axes[2].bar(flags.index, flags.values, color=["#E15759", "#B07AA1", "#9C755F"])
    axes[2].set_title("Design flags")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set(ylabel="Guides")
    hb = axes[3].hexbin(selected["n_cells_target"], selected["n_guides"], gridsize=35, mincnt=1)
    axes[3].set_title("Test support")
    boundaries, ticks = apply_density_scale(hb, "viridis")
    axes[3].set(xlabel="Cells per target-state test", ylabel="Guides")
    fig.colorbar(hb, ax=axes[3], boundaries=boundaries, ticks=ticks, label="Tests per hexagon")
    label_grid(axes)
    destination = args.figure_root / "S02"
    write_figure_data(destination, "A_guides_per_target.tsv", guide_counts.rename("guides").rename_axis("target_gene_id").reset_index())
    write_figure_data(destination, "B_guide_tss_distance.tsv", guides[["sgRNA", "distance_to_closest_target_tss"]])
    write_figure_data(destination, "C_design_flags.tsv", flags.rename("guides").rename_axis("category").reset_index())
    write_figure_data(destination, "D_test_support.tsv", selected[["target_contrast", "culture_condition", "n_cells_target", "n_guides"]])
    save_figure(fig, destination)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    guide_corr = selected["guide_correlation_all"].dropna()
    donor_corr = selected["donor_correlation_all_mean"].dropna()
    axes[0].hist(guide_corr, bins=35, color="#4C78A8", edgecolor="white")
    axes[0].set_title("Guide concordance")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set(xlabel="Between-guide correlation", ylabel="Tests")
    axes[1].hist(donor_corr, bins=35, color="#59A14F", edgecolor="white")
    axes[1].set_title("Donor concordance")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Mean between-donor correlation", ylabel="Tests")
    complete = selected[["guide_correlation_all", "donor_correlation_all_mean"]].dropna()
    hb = axes[2].hexbin(complete.iloc[:, 0], complete.iloc[:, 1], gridsize=28, mincnt=1)
    axes[2].set_title("Joint concordance")
    boundaries, ticks = apply_density_scale(hb, "magma")
    axes[2].set(xlabel="Guide correlation", ylabel="Donor correlation")
    fig.colorbar(hb, ax=axes[2], boundaries=boundaries, ticks=ticks, label="Tests per hexagon")
    state_data = [selected.loc[selected["culture_condition"] == state, "guide_correlation_all"].dropna() for state in colors]
    violin = axes[3].violinplot(state_data, showmedians=True, showextrema=False)
    axes[3].set_title("State distributions")
    for body, color in zip(violin["bodies"], colors.values(), strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.8)
    display_states = {"Rest": "Rest", "Stim8hr": "8 h", "Stim48hr": "48 h"}
    axes[3].set_xticks(
        range(1, len(colors) + 1), [display_states[state] for state in colors], rotation=0
    )
    axes[3].set(ylabel="Guide correlation")
    label_grid(axes)
    destination = args.figure_root / "S03"
    write_figure_data(destination, "A_guide_correlation.tsv", selected[["target_contrast", "culture_condition", "guide_correlation_all"]].dropna())
    write_figure_data(destination, "B_donor_correlation.tsv", selected[["target_contrast", "culture_condition", "donor_correlation_all_mean"]].dropna())
    write_figure_data(destination, "C_joint_reproducibility.tsv", selected[["target_contrast", "culture_condition", "guide_correlation_all", "donor_correlation_all_mean"]].dropna())
    write_figure_data(destination, "D_state_guide_correlation.tsv", selected[["target_contrast", "culture_condition", "guide_correlation_all"]].dropna())
    save_figure(fig, destination)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axes = axes.ravel()
    criterion_labels = {
        "all_rows": "All tests",
        "ontarget_significant": "Significant on-target effect",
        "adequate_target_expression": "Adequate target expression",
        "no_neighboring_gene_knockdown": "No neighboring-gene knockdown",
        "no_distal_offtarget": "No distal off-target flag",
        "minimum_guides": "At least two guides",
        "minimum_cells": "At least 200 cells",
        "complete_covariates": "Complete covariates",
        "primary_analysis_set": "Joint primary set",
    }
    cascade_display = cascade["criterion"].map(criterion_labels).fillna(cascade["criterion"])
    axes[0].barh(cascade_display, cascade["fraction_rows"], color="#4C78A8")
    axes[0].set_title("Inclusion cascade")
    axes[0].set(xlabel="Fraction of source tests", xlim=(0, 1.02))
    axes[1].hist(selected["ontarget_effect_size"], bins=45, color="#F28E2B", edgecolor="white")
    axes[1].set_title("On-target strength")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="On-target differential-expression z-score", ylabel="Tests")
    hb = axes[2].hexbin(np.abs(selected["ontarget_effect_size"]), np.log1p(selected["n_downstream"]), gridsize=40, mincnt=1)
    axes[2].set_title("Cis-trans burden")
    boundaries, ticks = apply_density_scale(hb, "viridis")
    axes[2].set(xlabel="Absolute on-target z-score", ylabel="log1p downstream genes")
    fig.colorbar(hb, ax=axes[2], boundaries=boundaries, ticks=ticks, label="Tests per hexagon")
    state_order = [state for state in colors if state in set(condition_rows["culture_condition"])]
    condition_rows = condition_rows.set_index("culture_condition").loc[state_order].reset_index()
    axes[3].bar(
        [display_states[state] for state in condition_rows["culture_condition"]],
        condition_rows["targets"],
        color=[colors[state] for state in condition_rows["culture_condition"]],
    )
    axes[3].set_title("State coverage")
    axes[3].tick_params(axis="x", rotation=0)
    axes[3].set(ylabel="Primary targets")
    label_grid(axes)
    destination = args.figure_root / "S04"
    write_figure_data(destination, "A_filter_cascade.tsv", cascade)
    write_figure_data(destination, "B_ontarget_zscore.tsv", selected[["target_contrast", "culture_condition", "ontarget_effect_size"]])
    write_figure_data(destination, "C_cis_trans_burden.tsv", selected[["target_contrast", "culture_condition", "ontarget_effect_size", "n_downstream"]])
    write_figure_data(destination, "D_state_targets.tsv", condition_rows)
    save_figure(fig, destination)

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
