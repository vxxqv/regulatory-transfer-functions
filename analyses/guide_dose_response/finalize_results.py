"""Finalize dose-response sensitivity and model-selection summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = np.asarray(values)[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def finalize(output: Path, seed: int, replicates: int, quality: list[str]) -> None:
    rng = np.random.default_rng(seed + 31)
    data = pd.read_csv(output / "eligible_guide_rows.csv")
    predictions = pd.read_parquet(output / "guide_held_out_predictions.parquet")
    sensitivity = []
    for state, block in data.groupby("culture_condition"):
        for dose in ["knockdown_fraction", "standardized_difference", "t_scaled"]:
            wide = block.pivot(index="pair_id", columns="guide_position", values=[dose, "response", *quality]).dropna()
            dx = wide[(dose, 2)].to_numpy() - wide[(dose, 1)].to_numpy()
            dy = wide[("response", 2)].to_numpy() - wide[("response", 1)].to_numpy()
            dq = np.column_stack([wide[(column, 2)].to_numpy() - wide[(column, 1)].to_numpy() for column in quality])
            dq = (dq - dq.mean(axis=0)) / np.where(dq.std(axis=0) == 0, 1, dq.std(axis=0))
            design = np.column_stack([dx, dq])
            beta = np.linalg.lstsq(design, dy, rcond=None)[0]
            standardized = beta[0] * np.std(dx) / max(np.std(dy), 1e-12)
            boot = []
            for _ in range(replicates):
                index = rng.integers(0, len(dx), len(dx))
                local = np.linalg.lstsq(design[index], dy[index], rcond=None)[0]
                boot.append(local[0] * np.std(dx[index]) / max(np.std(dy[index]), 1e-12))
            boot = np.asarray(boot)
            p = min(1.0, 2 * (1 + min((boot <= 0).sum(), (boot >= 0).sum())) / (len(boot) + 1))
            sensitivity.append({"culture_condition": state, "dose_definition": dose, "standardized_association": standardized, "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975), "bootstrap_p": p, "pairs": len(dx)})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity["q_value"] = bh(sensitivity["bootstrap_p"].to_numpy())
    sensitivity.to_csv(output / "measurement_error_sensitivity.csv", index=False)

    nonlinear = []
    for state, block in predictions.groupby("culture_condition"):
        linear = block[block.model == "linear"].copy()
        linear_error = linear.assign(error=(linear.observed - linear.predicted) ** 2).groupby("pair_id").error.mean()
        linear_r2 = r2_score(linear.observed, linear.predicted)
        for family in ["spline", "isotonic", "breakpoint", "hill"]:
            current = block[block.model == family].copy()
            current_error = current.assign(error=(current.observed - current.predicted) ** 2).groupby("pair_id").error.mean()
            shared = linear_error.index.intersection(current_error.index)
            p = float(wilcoxon(current_error.loc[shared], linear_error.loc[shared], alternative="less").pvalue)
            delta = r2_score(current.observed, current.predicted) - linear_r2
            boot = []
            pairs = np.asarray(shared)
            linear_indexed = linear.set_index("pair_id")
            current_indexed = current.set_index("pair_id")
            for _ in range(replicates):
                sample = rng.choice(pairs, len(pairs), replace=True)
                left = current_indexed.loc[sample]
                right = linear_indexed.loc[sample]
                boot.append(r2_score(left.observed, left.predicted) - r2_score(right.observed, right.predicted))
            nonlinear.append({"culture_condition": state, "model": family, "delta_r2_vs_linear": delta, "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975), "paired_error_p": p})
    nonlinear = pd.DataFrame(nonlinear)
    nonlinear["q_value"] = bh(nonlinear["paired_error_p"].to_numpy())
    nonlinear["supported"] = (nonlinear.delta_r2_vs_linear > 0) & (nonlinear.q_value < 0.05) & (nonlinear.ci_low > 0)
    nonlinear.to_csv(output / "nonlinear_comparisons.csv", index=False)

    permutation = pd.read_csv(output / "matched_guide_permutations.csv")
    summaries = []
    for state, block in permutation.groupby("culture_condition"):
        observed = block.observed_slope.iloc[0]
        p = (1 + (block.permuted_slope >= observed).sum()) / (len(block) + 1)
        summaries.append({"culture_condition": state, "observed_slope": observed, "null_median": block.permuted_slope.median(), "null_low": block.permuted_slope.quantile(0.025), "null_high": block.permuted_slope.quantile(0.975), "one_sided_p": p})
    summaries = pd.DataFrame(summaries)
    summaries["q_value"] = bh(summaries.one_sided_p.to_numpy())
    summaries.to_csv(output / "matched_guide_permutation_summary.csv", index=False)

    fold = pd.read_csv(output / "fold_metrics.csv")
    selection = fold.groupby(["culture_condition", "model"]).agg(mean_r2=("r2", "mean"), se_r2=("r2", lambda x: x.std(ddof=1) / np.sqrt(len(x)))).reset_index()
    records = []
    for state, block in selection.groupby("culture_condition"):
        best = block.loc[block.mean_r2.idxmax()]
        threshold = best.mean_r2 - best.se_r2
        for row in block.itertuples(index=False):
            records.append({**row._asdict(), "best_model": best.model, "within_one_se": row.mean_r2 >= threshold})
    pd.DataFrame(records).to_csv(output / "model_selection_uncertainty.csv", index=False)
    decisions = []
    for state, block in sensitivity.groupby("culture_condition"):
        decisions.append({"hypothesis": "positive guide-dose association", "culture_condition": state, "decision": "passed" if (block.ci_low > 0).all() else "unresolved", "basis": "all three pre-specified dose definitions"})
    for state, block in nonlinear.groupby("culture_condition"):
        passing = block[block.supported]
        decisions.append({"hypothesis": "nonlinear dose response", "culture_condition": state, "decision": "passed_small_effect" if len(passing) else "unresolved", "basis": ";".join(passing.model) if len(passing) else "no nonlinear family passed FDR and bootstrap criteria"})
    decisions.extend([
        {"hypothesis": "target-specific threshold or saturation", "culture_condition": "all", "decision": "underpowered", "basis": "two guide doses per eligible target-state"},
        {"hypothesis": "donor-held-out guide-dose replication", "culture_condition": "all", "decision": "unavailable", "basis": "no joint guide-by-donor effects"},
        {"hypothesis": "cis mediation", "culture_condition": "all", "decision": "not_tested", "basis": "identification assumptions failed"},
    ])
    pd.DataFrame(decisions).to_csv(output / "hypothesis_decisions.csv", index=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    finalize(root / "analyses/guide_dose_response/results", 1847, 1000, ["gc_fraction", "log1p_guide_cells", "log1p_tss_distance", "library_flag", "bidirectional_promoter", "secondary_alignment"])
