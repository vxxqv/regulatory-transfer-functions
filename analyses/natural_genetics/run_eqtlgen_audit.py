"""Retrospective audit correction for natural-genetic directional concordance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("Rest", "Stim8hr", "Stim48hr")
SUBSETS = ("primary", "bonferroni")
SCOPES = ("overall",) + CONDITIONS
METRICS = ("direction_agreement", "score_spearman")
FAMILY_SIZE = len(SUBSETS) * len(SCOPES) * len(METRICS)
COVARIATES = ("target_baseMean", "n_cells_target", "response_degree")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, default=Path("data/interim/gwt_vectors"))
    parser.add_argument("--eqtlgen", type=Path, default=Path("data/interim/eqtlgen"))
    parser.add_argument(
        "--output", type=Path, default=Path("analyses/natural_genetics/results_audit_corrected")
    )
    parser.add_argument(
        "--amendment", type=Path, default=Path("analyses/natural_genetics/audit_correction_01.yaml")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_palindromic(first: pd.Series, second: pd.Series) -> pd.Series:
    pair = first.astype(str).str.upper() + second.astype(str).str.upper()
    return pair.isin({"AT", "TA", "CG", "GC"})


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.unique(first[valid]).size < 2 or np.unique(second[valid]).size < 2:
        return np.nan
    return float(spearmanr(first[valid], second[valid]).statistic)


def fixed_family_bh(p_values: Iterable[float], family_size: int) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(values[finite], kind="mergesort")]
    ranked = values[order] * family_size / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def enumerate_bijections(targets: Iterable[str]) -> Iterator[dict[str, str]]:
    ordered = tuple(sorted(map(str, targets)))
    for permuted in permutations(ordered):
        yield dict(zip(ordered, permuted, strict=True))


def build_target_strata(
    rows: pd.DataFrame,
    response_degree: np.ndarray,
    conditions: Iterable[str] = CONDITIONS,
    minimum_targets: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = tuple(conditions)
    frame = rows.copy()
    if "response_row" not in frame:
        frame["response_row"] = np.arange(len(frame), dtype=int)
    if frame[["target_contrast", "culture_condition"]].duplicated().any():
        raise ValueError("Target-state rows must be unique")
    if len(response_degree) != len(frame):
        raise ValueError("Response degree must align with rows")
    frame["response_degree"] = np.asarray(response_degree, dtype=float)
    frame = frame.loc[frame["culture_condition"].isin(states)].copy()
    threshold_records: list[dict[str, object]] = []
    thresholds: dict[tuple[str, str], float] = {}
    for state in states:
        selected = frame.loc[frame["culture_condition"] == state]
        for covariate in COVARIATES:
            values = pd.to_numeric(selected[covariate], errors="coerce")
            median = float(values.median())
            if not np.isfinite(median):
                raise ValueError(f"Non-finite all-QC median for {state} {covariate}")
            thresholds[(state, covariate)] = median
            threshold_records.append(
                {
                    "condition": state,
                    "covariate": covariate,
                    "all_qc_rows": int(values.notna().sum()),
                    "median": median,
                    "low_rule": "value <= median",
                    "high_rule": "value > median",
                }
            )
    state_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for target, group in frame.groupby("target_contrast", sort=True):
        indexed = group.set_index("culture_condition", drop=False)
        availability = tuple(state for state in states if state in indexed.index)
        key_parts = ["states=" + ",".join(availability)]
        target_record: dict[str, object] = {
            "target": str(target),
            "state_availability": ",".join(availability),
        }
        for state in states:
            available = state in indexed.index
            target_record[f"available_{state}"] = bool(available)
            if not available:
                for covariate in COVARIATES:
                    target_record[f"{state}_{covariate}_bin"] = "NA"
                continue
            row = indexed.loc[state]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Duplicate target-state row for {target} {state}")
            bins: dict[str, str] = {}
            for covariate in COVARIATES:
                value = float(row[covariate])
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite stratum value for {target} {state} {covariate}")
                label = "low" if value <= thresholds[(state, covariate)] else "high"
                bins[covariate] = label
                target_record[f"{state}_{covariate}_bin"] = label
                key_parts.append(f"{state}:{covariate}:{label}")
            state_rows.append(
                {
                    "target": str(target),
                    "condition": state,
                    "response_row": int(row["response_row"]),
                    "target_baseMean": float(row["target_baseMean"]),
                    "n_cells_target": float(row["n_cells_target"]),
                    "response_degree": int(row["response_degree"]),
                    **{f"{name}_bin": label for name, label in bins.items()},
                }
            )
        target_record["stratum_key"] = "|".join(key_parts)
        target_rows.append(target_record)
    targets = pd.DataFrame(target_rows).sort_values("target", kind="mergesort").reset_index(drop=True)
    sizes = targets.groupby("stratum_key")["target"].transform("nunique").astype(int)
    targets["stratum_targets"] = sizes
    targets["permutable_stratum"] = sizes >= int(minimum_targets)
    target_to_stratum = targets.set_index("target")["stratum_key"]
    states_table = pd.DataFrame(state_rows)
    states_table["stratum_key"] = states_table["target"].map(target_to_stratum)
    states_table = states_table.sort_values(["target", "condition"], kind="mergesort").reset_index(drop=True)
    stratum_summary = (
        targets.groupby("stratum_key", as_index=False)
        .agg(distinct_targets=("target", "nunique"), state_availability=("state_availability", "first"))
        .sort_values(["state_availability", "stratum_key"], kind="mergesort")
        .reset_index(drop=True)
    )
    stratum_summary["permutable"] = stratum_summary["distinct_targets"] >= int(minimum_targets)
    return targets, states_table, pd.DataFrame(threshold_records), stratum_summary


def draw_bundle_mapping(
    targets: pd.DataFrame, rng: np.random.Generator, minimum_targets: int = 5
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    ordered = targets.sort_values(["stratum_key", "target"], kind="mergesort")
    for _, group in ordered.groupby("stratum_key", sort=True):
        members = group["target"].astype(str).sort_values(kind="mergesort").to_numpy()
        mapped = rng.permutation(members) if len(members) >= minimum_targets else members.copy()
        mapping.update(zip(members.tolist(), mapped.tolist(), strict=True))
    return mapping


def mapping_is_bundle_preserving(mapping: dict[str, str], targets: pd.DataFrame) -> bool:
    strata = targets.set_index("target")["stratum_key"].astype(str).to_dict()
    if set(mapping) != set(strata) or len(set(mapping.values())) != len(mapping):
        return False
    return all(strata[source] == strata[destination] for source, destination in mapping.items())


def build_association_universe(
    rows: pd.DataFrame,
    genes: pd.DataFrame,
    cis: pd.DataFrame,
    trans: pd.DataFrame,
    allele_orientation,
    conditions: Iterable[str] = CONDITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = rows.copy()
    if "response_row" not in frame:
        frame["response_row"] = np.arange(len(frame), dtype=int)
    response_lookup = frame.set_index(["target_contrast", "culture_condition"])["response_row"]
    if not response_lookup.index.is_unique:
        raise ValueError("Response lookup is not unique")
    if genes["feature_id"].astype(str).duplicated().any():
        raise ValueError("Gene feature identifiers are not unique")
    gene_lookup = pd.Series(
        np.arange(len(genes), dtype=int), index=genes["feature_id"].astype(str)
    ).to_dict()
    merged = cis.merge(trans, on="SNP", suffixes=("_cis", "_trans"), validate="one_to_many")
    counts: list[dict[str, object]] = [
        {"stage": "cis_trans_merge", "rows": int(len(merged)), "snps": int(merged["SNP"].nunique())}
    ]
    merged["gene_column"] = merged["Gene_trans"].astype(str).map(gene_lookup)
    merged = merged.loc[merged["gene_column"].notna()].copy()
    merged["gene_column"] = merged["gene_column"].astype(int)
    counts.append(
        {"stage": "measured_trans_gene", "rows": int(len(merged)), "snps": int(merged["SNP"].nunique())}
    )
    palindromic = is_palindromic(merged["AssessedAllele_cis"], merged["OtherAllele_cis"]) | is_palindromic(
        merged["AssessedAllele_trans"], merged["OtherAllele_trans"]
    )
    counts.append(
        {
            "stage": "excluded_unresolved_palindromic",
            "rows": int(palindromic.sum()),
            "snps": int(merged.loc[palindromic, "SNP"].nunique()),
        }
    )
    merged = merged.loc[~palindromic].copy()
    orientation = np.array(
        [
            allele_orientation(a, b, c, d)
            for a, b, c, d in zip(
                merged["AssessedAllele_cis"],
                merged["OtherAllele_cis"],
                merged["AssessedAllele_trans"],
                merged["OtherAllele_trans"],
                strict=True,
            )
        ],
        dtype=int,
    )
    merged["allele_orientation"] = orientation
    unresolved = merged["allele_orientation"] == 0
    counts.append(
        {
            "stage": "excluded_other_unresolved_orientation",
            "rows": int(unresolved.sum()),
            "snps": int(merged.loc[unresolved, "SNP"].nunique()),
        }
    )
    merged = merged.loc[~unresolved].copy()
    records: list[pd.DataFrame] = []
    for condition in conditions:
        current = merged.copy()
        keys = pd.MultiIndex.from_arrays(
            [current["Gene_cis"].astype(str), np.repeat(condition, len(current))]
        )
        current["response_row_identity"] = response_lookup.reindex(keys).to_numpy()
        current = current.loc[current["response_row_identity"].notna()].copy()
        current["response_row_identity"] = current["response_row_identity"].astype(int)
        current["condition"] = str(condition)
        records.append(current)
    universe = pd.concat(records, ignore_index=True)
    universe["observed_trans_z"] = (
        universe["Zscore_trans"].astype(float) * universe["allele_orientation"].astype(float)
    )
    universe = universe.sort_values(
        ["SNPChr_cis", "SNPPos_cis", "SNP", "Gene_cis", "Gene_trans", "condition"],
        kind="mergesort",
    ).reset_index(drop=True)
    universe.insert(0, "association_id", np.arange(len(universe), dtype=int))
    counts.append(
        {
            "stage": "pre_observed_effect_selection_universe",
            "rows": int(len(universe)),
            "snps": int(universe["SNP"].nunique()),
        }
    )
    return universe, pd.DataFrame(counts)


def mapped_effects(
    universe: pd.DataFrame,
    mapping: dict[str, str],
    state_rows: pd.DataFrame,
    matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = state_rows.set_index(["target", "condition"])["response_row"]
    mapped_target = universe["Gene_cis"].astype(str).map(mapping)
    if mapped_target.isna().any():
        raise ValueError("A mediator is missing from the target mapping")
    keys = pd.MultiIndex.from_arrays([mapped_target, universe["condition"].astype(str)])
    response_rows = lookup.reindex(keys).to_numpy()
    if pd.isna(response_rows).any():
        raise ValueError("A mapped bundle lacks a required state")
    response_rows = response_rows.astype(int)
    gene_columns = universe["gene_column"].to_numpy(dtype=int)
    effects = np.asarray(matrix[response_rows, gene_columns]).ravel().astype(float)
    fixed = (
        np.isfinite(universe["Zscore_cis"].to_numpy(dtype=float))
        & (universe["Zscore_cis"].to_numpy(dtype=float) != 0)
        & np.isfinite(universe["observed_trans_z"].to_numpy(dtype=float))
        & (universe["observed_trans_z"].to_numpy(dtype=float) != 0)
    )
    eligible = fixed & np.isfinite(effects) & (effects != 0)
    return effects, eligible, response_rows


def slot_mask(
    universe: pd.DataFrame,
    eligible: np.ndarray,
    subset: str,
    scope: str,
    additional: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.asarray(eligible, dtype=bool).copy()
    if subset == "bonferroni":
        mask &= universe["BonferroniP_trans"].to_numpy(dtype=float) <= 0.05
    elif subset != "primary":
        raise ValueError(f"Unknown subset: {subset}")
    if scope != "overall":
        mask &= universe["condition"].astype(str).to_numpy() == scope
    if additional is not None:
        mask &= np.asarray(additional, dtype=bool)
    return mask


def calculate_statistic(
    universe: pd.DataFrame,
    effects: np.ndarray,
    eligible: np.ndarray,
    subset: str,
    scope: str,
    metric: str,
    additional: np.ndarray | None = None,
) -> tuple[float, int]:
    mask = slot_mask(universe, eligible, subset, scope, additional)
    denominator = int(mask.sum())
    if denominator == 0:
        return np.nan, denominator
    cis_sign = np.sign(universe["Zscore_cis"].to_numpy(dtype=float)[mask])
    score = -cis_sign * effects[mask]
    observed_z = universe["observed_trans_z"].to_numpy(dtype=float)[mask]
    if metric == "direction_agreement":
        return float(np.mean(np.sign(score) == np.sign(observed_z))), denominator
    if metric == "score_spearman":
        return safe_spearman(score, observed_z), denominator
    raise ValueError(f"Unknown metric: {metric}")


def slots() -> list[tuple[str, str, str]]:
    return [(subset, scope, metric) for subset in SUBSETS for scope in SCOPES for metric in METRICS]


def chromosome_cluster_bootstrap(
    universe: pd.DataFrame,
    effects: np.ndarray,
    eligible: np.ndarray,
    subset: str,
    scope: str,
    metric: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    base = slot_mask(universe, eligible, subset, scope)
    chromosomes = np.sort(universe.loc[base, "SNPChr_cis"].unique())
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    if len(chromosomes) == 0:
        return pd.DataFrame(
            {
                "replicate": np.arange(replicates, dtype=int),
                "estimate": np.nan,
                "denominator": 0,
                "source_chromosome_draws": 0,
                "distinct_source_chromosomes": 0,
            }
        )
    chromosome_values = universe["SNPChr_cis"].to_numpy()
    for replicate in range(replicates):
        drawn = rng.choice(chromosomes, size=len(chromosomes), replace=True)
        indices = np.concatenate([np.flatnonzero(base & (chromosome_values == chrom)) for chrom in drawn])
        if metric == "direction_agreement":
            cis_sign = np.sign(universe["Zscore_cis"].to_numpy(dtype=float)[indices])
            score = -cis_sign * effects[indices]
            observed = universe["observed_trans_z"].to_numpy(dtype=float)[indices]
            estimate = float(np.mean(np.sign(score) == np.sign(observed))) if len(indices) else np.nan
        else:
            cis_sign = np.sign(universe["Zscore_cis"].to_numpy(dtype=float)[indices])
            estimate = safe_spearman(
                -cis_sign * effects[indices],
                universe["observed_trans_z"].to_numpy(dtype=float)[indices],
            )
        records.append(
            {
                "replicate": replicate,
                "estimate": estimate,
                "denominator": int(len(indices)),
                "source_chromosome_draws": int(len(drawn)),
                "distinct_source_chromosomes": int(np.unique(drawn).size),
            }
        )
    return pd.DataFrame(records)


def leave_one_chromosome_out(
    universe: pd.DataFrame,
    identity_effect: np.ndarray,
    identity_eligible: np.ndarray,
    permutation_effects: np.ndarray,
    permutation_eligible: np.ndarray,
    full_summary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    chromosomes = np.sort(universe["SNPChr_cis"].unique())
    chromosome_values = universe["SNPChr_cis"].to_numpy()
    full_lookup = full_summary.set_index(["subset", "scope", "metric"])["estimate"].to_dict()
    for chromosome in chromosomes:
        retained = chromosome_values != chromosome
        for subset, scope, metric in slots():
            estimate, denominator = calculate_statistic(
                universe, identity_effect, identity_eligible, subset, scope, metric, retained
            )
            null_values = np.empty(len(permutation_effects), dtype=float)
            null_denominators = np.empty(len(permutation_effects), dtype=int)
            for replicate in range(len(permutation_effects)):
                null_values[replicate], null_denominators[replicate] = calculate_statistic(
                    universe,
                    permutation_effects[replicate],
                    permutation_eligible[replicate],
                    subset,
                    scope,
                    metric,
                    retained,
                )
            finite = np.isfinite(null_values)
            p_value = (
                float((1 + np.sum(null_values[finite] >= estimate)) / (1 + finite.sum()))
                if np.isfinite(estimate) and finite.any()
                else np.nan
            )
            full = float(full_lookup[(subset, scope, metric)])
            records.append(
                {
                    "excluded_source_chromosome": int(chromosome),
                    "subset": subset,
                    "scope": scope,
                    "metric": metric,
                    "estimate": estimate,
                    "full_estimate": full,
                    "difference_from_full": estimate - full if np.isfinite(estimate) and np.isfinite(full) else np.nan,
                    "denominator": denominator,
                    "source_chromosomes_retained": int(
                        universe.loc[retained & slot_mask(universe, identity_eligible, subset, scope), "SNPChr_cis"].nunique()
                    ),
                    "null_finite_replicates": int(finite.sum()),
                    "null_denominator_median": float(np.median(null_denominators[finite])) if finite.any() else np.nan,
                    "permutation_p_upper": p_value,
                    "strata_frozen": True,
                    "permutation_mappings_frozen": True,
                }
            )
    return pd.DataFrame(records)


def verify_amendment(path: Path) -> dict[str, object]:
    amendment = yaml.safe_load(path.read_text(encoding="utf-8"))
    for relative, expected in amendment["inputs"].items():
        observed = sha256_file(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"Input hash mismatch for {relative}")
    old_root = ROOT / "analyses/natural_genetics/results"
    for name, expected in amendment["original_outputs"]["files"].items():
        observed = sha256_file(old_root / name)
        if observed != expected:
            raise RuntimeError(f"Original output hash mismatch for {name}")
    return amendment


def main() -> None:
    args = parse_args()
    args.vectors = (ROOT / args.vectors).resolve() if not args.vectors.is_absolute() else args.vectors
    args.eqtlgen = (ROOT / args.eqtlgen).resolve() if not args.eqtlgen.is_absolute() else args.eqtlgen
    args.output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    args.amendment = (ROOT / args.amendment).resolve() if not args.amendment.is_absolute() else args.amendment
    amendment = verify_amendment(args.amendment)
    design = amendment["design"]
    seed = int(design["seed"])
    n_permutations = int(design["permutations"])
    n_bootstrap = int(design["chromosome_cluster_bootstrap_replicates"])
    minimum_targets = int(design["minimum_distinct_targets_to_permute"])
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from src.models.natural_genetics import allele_orientation

    rows = pd.read_parquet(args.vectors / "rows.parquet").reset_index(drop=True)
    rows["response_row"] = np.arange(len(rows), dtype=int)
    genes = pd.read_parquet(args.vectors / "genes.parquet").reset_index(drop=True)
    matrix = sparse.load_npz(args.vectors / "normalized_significant_logfc.npz").tocsr()
    cis = pd.read_parquet(args.eqtlgen / "cis_mediators.parquet")
    trans = pd.read_parquet(args.eqtlgen / "trans_associations.parquet")
    response_degree = np.asarray((matrix != 0).sum(axis=1)).ravel()

    target_table, state_rows, thresholds, stratum_summary = build_target_strata(
        rows, response_degree, CONDITIONS, minimum_targets
    )
    universe, exclusion_flow = build_association_universe(
        rows, genes, cis, trans, allele_orientation, CONDITIONS
    )
    identity = {target: target for target in target_table["target"].astype(str)}
    identity_effect, identity_eligible, identity_rows = mapped_effects(
        universe, identity, state_rows, matrix
    )
    universe["identity_effect"] = identity_effect
    universe["identity_eligible"] = identity_eligible
    universe["identity_response_row"] = identity_rows
    universe["predicted_transfer_score"] = -np.sign(universe["Zscore_cis"]) * identity_effect
    universe["predicted_direction"] = np.sign(universe["predicted_transfer_score"])
    universe["observed_direction"] = np.sign(universe["observed_trans_z"])
    universe["direction_match"] = (
        universe["predicted_direction"] == universe["observed_direction"]
    ).where(universe["identity_eligible"])

    observed_records: list[dict[str, object]] = []
    for subset, scope, metric in slots():
        estimate, denominator = calculate_statistic(
            universe, identity_effect, identity_eligible, subset, scope, metric
        )
        mask = slot_mask(universe, identity_eligible, subset, scope)
        observed_records.append(
            {
                "subset": subset,
                "scope": scope,
                "metric": metric,
                "estimate": estimate,
                "pairs": denominator,
                "snps": int(universe.loc[mask, "SNP"].nunique()),
                "mediators": int(universe.loc[mask, "Gene_cis"].nunique()),
                "trans_genes": int(universe.loc[mask, "Gene_trans"].nunique()),
                "source_chromosomes": int(universe.loc[mask, "SNPChr_cis"].nunique()),
            }
        )
    observed = pd.DataFrame(observed_records)

    rng = np.random.default_rng(seed)
    permutation_effects = np.empty((n_permutations, len(universe)), dtype=np.float32)
    permutation_eligible = np.empty((n_permutations, len(universe)), dtype=bool)
    null_records: list[dict[str, object]] = []
    mapping_records: list[dict[str, object]] = []
    mediator_targets = sorted(universe["Gene_cis"].astype(str).unique())
    target_meta = target_table.set_index("target")[["stratum_key", "permutable_stratum"]]
    for replicate in range(n_permutations):
        mapping = draw_bundle_mapping(target_table, rng, minimum_targets)
        if not mapping_is_bundle_preserving(mapping, target_table):
            raise RuntimeError(f"Invalid target-bundle mapping at replicate {replicate}")
        effects, eligible, _ = mapped_effects(universe, mapping, state_rows, matrix)
        permutation_effects[replicate] = effects.astype(np.float32)
        permutation_eligible[replicate] = eligible
        for mediator in mediator_targets:
            mapped = mapping[mediator]
            mapping_records.append(
                {
                    "replicate": replicate,
                    "mediator": mediator,
                    "mapped_target": mapped,
                    "stratum_key": target_meta.loc[mediator, "stratum_key"],
                    "permutable_stratum": bool(target_meta.loc[mediator, "permutable_stratum"]),
                    "fixed_point": mediator == mapped,
                }
            )
        for subset, scope, metric in slots():
            statistic, denominator = calculate_statistic(
                universe, effects, eligible, subset, scope, metric
            )
            null_records.append(
                {
                    "replicate": replicate,
                    "subset": subset,
                    "scope": scope,
                    "metric": metric,
                    "statistic": statistic,
                    "denominator": denominator,
                }
            )
    null_table = pd.DataFrame(null_records)

    bootstrap_tables: list[pd.DataFrame] = []
    summary = observed.copy()
    summary["bootstrap_ci_low"] = np.nan
    summary["bootstrap_ci_high"] = np.nan
    summary["bootstrap_finite_replicates"] = 0
    summary["null_median"] = np.nan
    summary["null_denominator_min"] = np.nan
    summary["null_denominator_median"] = np.nan
    summary["null_denominator_max"] = np.nan
    summary["null_finite_replicates"] = 0
    summary["permutation_p_upper"] = np.nan
    for index, row in summary.iterrows():
        subset, scope, metric = row["subset"], row["scope"], row["metric"]
        bootstrap = chromosome_cluster_bootstrap(
            universe,
            identity_effect,
            identity_eligible,
            subset,
            scope,
            metric,
            n_bootstrap,
            seed,
        )
        bootstrap.insert(0, "metric", metric)
        bootstrap.insert(0, "scope", scope)
        bootstrap.insert(0, "subset", subset)
        bootstrap_tables.append(bootstrap)
        finite_boot = bootstrap["estimate"].to_numpy(dtype=float)
        finite_boot = finite_boot[np.isfinite(finite_boot)]
        if finite_boot.size:
            summary.loc[index, ["bootstrap_ci_low", "bootstrap_ci_high"]] = np.quantile(
                finite_boot, [0.025, 0.975]
            )
        summary.loc[index, "bootstrap_finite_replicates"] = int(finite_boot.size)
        null = null_table.loc[
            (null_table["subset"] == subset)
            & (null_table["scope"] == scope)
            & (null_table["metric"] == metric)
        ]
        values = null["statistic"].to_numpy(dtype=float)
        denominators = null["denominator"].to_numpy(dtype=int)
        finite = np.isfinite(values)
        summary.loc[index, "null_denominator_min"] = int(denominators.min())
        summary.loc[index, "null_denominator_median"] = float(np.median(denominators))
        summary.loc[index, "null_denominator_max"] = int(denominators.max())
        summary.loc[index, "null_finite_replicates"] = int(finite.sum())
        if finite.any():
            summary.loc[index, "null_median"] = float(np.median(values[finite]))
            if finite.all() and np.isfinite(row["estimate"]):
                summary.loc[index, "permutation_p_upper"] = float(
                    (1 + np.sum(values >= row["estimate"])) / (1 + len(values))
                )
    summary["bh_family_slots"] = FAMILY_SIZE
    summary["bh_available_slots"] = int(summary["permutation_p_upper"].notna().sum())
    summary["permutation_q_fixed_16"] = fixed_family_bh(
        summary["permutation_p_upper"], FAMILY_SIZE
    )
    summary["available"] = summary["estimate"].notna() & summary["permutation_p_upper"].notna()
    summary["unavailable_reason"] = np.where(
        summary["estimate"].isna(),
        "observed_statistic_not_estimable",
        np.where(
            summary["null_finite_replicates"].lt(n_permutations),
            "one_or_more_scheduled_null_replicates_undefined",
            np.where(summary["permutation_p_upper"].isna(), "null_not_estimable", ""),
        ),
    )

    loco = leave_one_chromosome_out(
        universe,
        identity_effect,
        identity_eligible,
        permutation_effects,
        permutation_eligible,
        summary,
    )
    eligible_pairs = universe.loc[universe["identity_eligible"]].copy()
    bootstrap_table = pd.concat(bootstrap_tables, ignore_index=True)
    mappings = pd.DataFrame(mapping_records)

    thresholds.to_csv(args.output / "stratum_thresholds.csv", index=False)
    target_table.to_csv(args.output / "target_strata.csv", index=False)
    state_rows.to_parquet(args.output / "target_state_rows.parquet", index=False)
    stratum_summary.to_csv(args.output / "stratum_summary.csv", index=False)
    exclusion_flow.to_csv(args.output / "input_exclusion_flow.csv", index=False)
    universe.to_parquet(args.output / "association_universe.parquet", index=False)
    eligible_pairs.to_parquet(args.output / "directional_pairs.parquet", index=False)
    mappings.to_parquet(args.output / "permutation_mediator_mappings.parquet", index=False)
    null_table.to_parquet(args.output / "matched_null.parquet", index=False)
    bootstrap_table.to_parquet(args.output / "chromosome_cluster_bootstrap.parquet", index=False)
    summary.to_csv(args.output / "hypothesis_tests.csv", index=False)
    loco.to_csv(args.output / "leave_one_source_chromosome_out.csv", index=False)

    primary = summary.set_index(["subset", "scope", "metric"])
    metrics = {
        "correction_id": amendment["correction_id"],
        "status": "retrospective_audit_correction",
        "seed": seed,
        "permutations": n_permutations,
        "chromosome_cluster_bootstrap_replicates": n_bootstrap,
        "pre_effect_selection_rows": int(len(universe)),
        "identity_eligible_pairs": int(identity_eligible.sum()),
        "identity_ineligible_zero_or_nonfinite_pairs": int((~identity_eligible).sum()),
        "unresolved_palindromic_rows_excluded_before_state_expansion": int(
            exclusion_flow.loc[
                exclusion_flow["stage"] == "excluded_unresolved_palindromic", "rows"
            ].iloc[0]
        ),
        "unresolved_palindromic_snps_excluded": int(
            exclusion_flow.loc[
                exclusion_flow["stage"] == "excluded_unresolved_palindromic", "snps"
            ].iloc[0]
        ),
        "targets_in_mapping_universe": int(target_table["target"].nunique()),
        "permutable_strata": int(stratum_summary["permutable"].sum()),
        "small_identity_strata": int((~stratum_summary["permutable"]).sum()),
        "hypothesis_family_slots": FAMILY_SIZE,
        "hypothesis_available_slots": int(summary["available"].sum()),
        "primary_overall_direction_agreement": float(
            primary.loc[("primary", "overall", "direction_agreement"), "estimate"]
        ),
        "primary_overall_direction_ci_95": [
            float(primary.loc[("primary", "overall", "direction_agreement"), "bootstrap_ci_low"]),
            float(primary.loc[("primary", "overall", "direction_agreement"), "bootstrap_ci_high"]),
        ],
        "primary_overall_direction_permutation_p": float(
            primary.loc[("primary", "overall", "direction_agreement"), "permutation_p_upper"]
        ),
        "primary_overall_direction_q_fixed_16": float(
            primary.loc[("primary", "overall", "direction_agreement"), "permutation_q_fixed_16"]
        ),
        "primary_overall_score_spearman": float(
            primary.loc[("primary", "overall", "score_spearman"), "estimate"]
        ),
        "primary_overall_score_permutation_p": float(
            primary.loc[("primary", "overall", "score_spearman"), "permutation_p_upper"]
        ),
        "primary_overall_score_q_fixed_16": float(
            primary.loc[("primary", "overall", "score_spearman"), "permutation_q_fixed_16"]
        ),
    }
    (args.output / "natural_genetics_results.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "audit.json"
    }
    audit = {
        "correction_id": amendment["correction_id"],
        "amendment_sha256": sha256_file(args.amendment),
        "script_sha256": sha256_file(Path(__file__)),
        "original_outputs_verified_unchanged": True,
        "input_hashes_verified": True,
        "row_order_rule": "stable lexical target and stratum ordering before seeded bijections",
        "mapping_bundle_integrity_all_replicates": True,
        "one_mapping_per_mediator_all_associations_states": True,
        "identity_and_permutation_eligibility_symmetric": True,
        "null_replicate_denominators_stored": True,
        "available_slots_require_all_scheduled_null_replicates_finite": True,
        "loco_mappings_and_strata_frozen": True,
        "output_hashes": output_hashes,
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
