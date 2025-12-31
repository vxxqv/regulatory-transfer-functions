"""Run state-resolved regulatory verification on frozen study outputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyliftover import LiftOver
from scipy import sparse
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("REGULATORY_SOURCE_ROOT", ROOT)).resolve()
OUTPUT = ROOT / "analyses/molecular_cascade/results"
STATE_ORDER = ["Rest", "Stim8hr", "Stim48hr"]


def verify_freeze() -> list[dict]:
    manifest = json.loads((ROOT / "analyses/molecular_cascade/freeze_manifest.json").read_text())
    revision_path = ROOT / "analyses/molecular_cascade/protocol_revision.json"
    revision = json.loads(revision_path.read_text()) if revision_path.exists() else None
    verified = []
    for artifact in manifest["artifacts"]:
        path = SOURCE / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            if artifact["path"] != "docs/protocol.md" or revision is None:
                raise ValueError(f"Frozen input changed: {artifact['path']}")
            frozen = subprocess.check_output(
                ["git", "show", f"{revision['frozen_protocol_commit']}:docs/protocol.md"], cwd=ROOT
            )
            frozen_hash = hashlib.sha256(frozen).hexdigest()
            if frozen_hash != artifact["sha256"]:
                raise ValueError("Historical protocol does not match the frozen hash")
            verified.append({"path": artifact["path"], "sha256": frozen_hash, "match": True, "source": "historical_git_object", "current_sha256": actual})
            continue
        verified.append({"path": artifact["path"], "sha256": actual, "match": True, "source": "current_file"})
    return verified


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    local = values[valid]
    order = np.argsort(local)
    ranked = local[order]
    adjusted = np.minimum.accumulate((ranked * len(local) / np.arange(1, len(local) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def read_bed(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    table = pd.read_csv(path, sep="\t", compression="gzip", comment="#", header=None, usecols=[0, 1, 2])
    table.columns = ["chromosome", "start", "end"]
    table["chromosome"] = table["chromosome"].astype(str).str.removeprefix("chr")
    result = {}
    for chromosome, block in table.groupby("chromosome"):
        result[str(chromosome)] = (
            np.sort(block["start"].to_numpy(dtype=np.int64)),
            np.sort(block["end"].to_numpy(dtype=np.int64)),
        )
    return result


def overlap_count(index: dict[str, tuple[np.ndarray, np.ndarray]], chromosome: str, start: int, end: int) -> int:
    values = index.get(str(chromosome).removeprefix("chr"))
    if values is None or not np.isfinite(start) or not np.isfinite(end):
        return 0
    starts, ends = values
    return max(0, int(np.searchsorted(starts, int(end), side="left") - np.searchsorted(ends, int(start), side="right")))


def load_curated(path: Path, measured: set[str]) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    data["source_gene"] = data["source_genesymbol"].astype(str).str.upper()
    data["response_gene"] = data["target_genesymbol"].astype(str).str.upper()
    source = data["sources"].fillna("").astype(str)
    data["dorothea_a"] = source.str.contains(r"(?:^|;)DoRothEA-A(?:_|;|$)", regex=True)
    data["dorothea_b"] = source.str.contains(r"(?:^|;)DoRothEA-B(?:_|;|$)", regex=True)
    data["dorothea_any"] = source.str.contains("DoRothEA", regex=False)
    data["trrust"] = source.str.contains(r"(?:^|;)TRRUST(?:_|;|$)", regex=True)
    data["regnetwork"] = source.str.contains("RegNetwork", regex=False)
    data["collectri"] = source.str.contains("CollecTRI", regex=False)
    stimulation = data["consensus_stimulation"].fillna(False).astype(bool)
    inhibition = data["consensus_inhibition"].fillna(False).astype(bool)
    data["direction"] = np.select([stimulation & ~inhibition, inhibition & ~stimulation], [1, -1], default=0)
    data = data[data["source_gene"].ne(data["response_gene"]) & data["response_gene"].isin(measured)].copy()
    records = []
    for (source_gene, response_gene), block in data.groupby(["source_gene", "response_gene"], sort=False):
        directions = set(block.loc[block["direction"].ne(0), "direction"].astype(int))
        direction_conflict = len(directions) > 1
        direction = next(iter(directions)) if len(directions) == 1 else 0
        flags = {name: bool(block[name].any()) for name in ["dorothea_a", "dorothea_b", "dorothea_any", "trrust", "regnetwork", "collectri"]}
        resources = [name for name, column in [("DoRothEA-A", "dorothea_a"), ("DoRothEA-B", "dorothea_b"), ("TRRUST", "trrust"), ("RegNetwork", "regnetwork"), ("CollecTRI", "collectri")] if flags[column]]
        records.append({
            "source_gene": source_gene,
            "response_gene": response_gene,
            "curated_direction": direction,
            "direction_conflict": direction_conflict,
            "primary_curated": flags["dorothea_a"] or flags["trrust"] or flags["regnetwork"],
            "sensitivity_curated": flags["dorothea_any"] or flags["collectri"],
            "resources": ";".join(resources),
            **flags,
        })
    return pd.DataFrame(records).sort_values(["source_gene", "response_gene"], kind="stable").reset_index(drop=True)


def load_motifs(path: Path, measured: set[str], minimum: int, maximum: int) -> dict[str, set[str]]:
    motifs: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or "(human)" not in fields[0].lower():
                continue
            regulator = fields[0].split(" (")[0].strip().upper()
            targets = {value.upper() for value in fields[2:] if value.upper() in measured}
            if minimum <= len(targets) <= maximum:
                motifs[regulator].update(targets)
    return dict(motifs)


def build_gene_table(genes: pd.DataFrame, config: dict, beds: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> pd.DataFrame:
    attributes = pd.read_csv(SOURCE / "work/regulatory_inputs/ensembl_gene_attributes.tsv", sep="\t")
    attributes = attributes.rename(columns={
        "Gene stable ID": "feature_id",
        "Gene name": "attribute_gene_name",
        "Chromosome/scaffold name": "chromosome",
        "Gene start (bp)": "gene_start",
        "Gene end (bp)": "gene_end",
        "Strand": "strand",
        "Gene % GC content": "gene_gc_percent",
    })
    attributes["feature_id"] = attributes["feature_id"].astype(str).str.split(".").str[0]
    attributes = attributes.drop_duplicates("feature_id")
    table = genes.copy()
    table["feature_id"] = table["feature_id"].astype(str).str.split(".").str[0]
    table["gene_name"] = table["gene_name"].astype(str).str.upper()
    table = table.merge(attributes[["feature_id", "chromosome", "gene_start", "gene_end", "strand", "gene_gc_percent"]], on="feature_id", how="left", validate="one_to_one")
    table["chromosome"] = table["chromosome"].astype(str).str.removeprefix("chr")
    table["tss"] = np.where(table["strand"].eq(1), table["gene_start"], table["gene_end"]) - 1
    upstream = int(config["chromatin"]["promoter_upstream_bp"])
    downstream = int(config["chromatin"]["promoter_downstream_bp"])
    table["promoter_start"] = np.maximum(0, np.where(table["strand"].eq(1), table["tss"] - upstream, table["tss"] - downstream))
    table["promoter_end"] = np.where(table["strand"].eq(1), table["tss"] + downstream, table["tss"] + upstream)
    for label, bed in beds.items():
        table[f"{label}_peak_count"] = [overlap_count(bed, chromosome, start, end) for chromosome, start, end in table[["chromosome", "promoter_start", "promoter_end"]].itertuples(index=False, name=None)]
    table["gene_interval_length"] = table["gene_end"] - table["gene_start"] + 1
    return table


def lift_interval(converter: LiftOver, chromosome: str, start: int, end: int) -> tuple[str | None, int | None, int | None]:
    chromosome = str(chromosome)
    chromosome = chromosome if chromosome.startswith("chr") else f"chr{chromosome}"
    first = converter.convert_coordinate(chromosome, int(start))
    last = converter.convert_coordinate(chromosome, max(int(start), int(end) - 1))
    if not first or not last:
        return None, None, None
    first_best = max(first, key=lambda value: value[3])
    last_best = max(last, key=lambda value: value[3])
    if first_best[0] != last_best[0] or first_best[2] != last_best[2] or first_best[0] != chromosome:
        return None, None, None
    mapped = sorted([int(first_best[1]), int(last_best[1])])
    return first_best[0].removeprefix("chr"), mapped[0], mapped[1] + 1


def prepare_pchic(config: dict, measured: set[str], beds: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(SOURCE / "work/cd4_pchic_interactions.tsv.gz", sep=r"\s+", compression="gzip")
    source_rows = len(source)
    source["gene"] = source["gene"].astype(str).str.upper()
    threshold = float(config["chromatin"]["pchic_primary_call_threshold"])
    source = source[source["gene"].isin(measured) & ((source["Total_CD4_Activated"] > threshold) | (source["Total_CD4_NonActivated"] > threshold))].copy()
    converter = LiftOver(str(SOURCE / "work/regulatory_inputs/hg19ToHg38.over.chain.gz"))
    records = []
    failed = 0
    for row in source.itertuples(index=False):
        bait = lift_interval(converter, row.baitChr, row.baitStart, row.baitStart + row.baitLength)
        prey = lift_interval(converter, row.oeChr, row.oeStart, row.oeStart + row.oeLength)
        if bait[0] is None or prey[0] is None:
            failed += 1
            continue
        records.append({
            "gene": row.gene,
            "bait_chromosome": bait[0],
            "bait_start": bait[1],
            "bait_end": bait[2],
            "prey_chromosome": prey[0],
            "prey_start": prey[1],
            "prey_end": prey[2],
            "chicago_activated": row.Total_CD4_Activated,
            "chicago_nonactivated": row.Total_CD4_NonActivated,
            "differential_logfc": row.logFC,
            "differential_fdr": row.FDR,
            "prey_atac_count": overlap_count(beds["atac"], prey[0], prey[1], prey[2]),
            "prey_h3k27ac_count": overlap_count(beds["h3k27ac"], prey[0], prey[1], prey[2]),
            "prey_ctcf_count": overlap_count(beds["ctcf"], prey[0], prey[1], prey[2]),
            "bait_ctcf_count": overlap_count(beds["ctcf"], bait[0], bait[1], bait[2]),
        })
    links = pd.DataFrame(records)
    links["active_both"] = links["prey_atac_count"].gt(0) & links["prey_h3k27ac_count"].gt(0)
    links["active_either"] = links["prey_atac_count"].gt(0) | links["prey_h3k27ac_count"].gt(0)
    audit = {
        "source_rows": int(source_rows),
        "eligible_called_rows": int(len(source)),
        "lifted_rows": int(len(links)),
        "failed_liftover_rows": int(failed),
        "liftover_rate": float(len(links) / len(source)) if len(source) else np.nan,
    }
    return links, audit


def pchic_support(links: pd.DataFrame, config: dict, threshold: float | None = None, active_rule: str = "atac_and_h3k27ac") -> pd.DataFrame:
    threshold = float(threshold if threshold is not None else config["chromatin"]["pchic_primary_call_threshold"])
    records = []
    active_column = "active_both" if active_rule == "atac_and_h3k27ac" else "active_either"
    for state in STATE_ORDER:
        score = "chicago_nonactivated" if state == "Rest" else "chicago_activated"
        block = links[links[score].gt(threshold)].copy()
        block["physical_active"] = block[active_column]
        block["ctcf_enhancer"] = block["physical_active"] & block["prey_ctcf_count"].gt(0)
        summary = block.groupby("gene", as_index=False).agg(
            called_links=(score, "size"),
            active_links=("physical_active", "sum"),
            ctcf_enhancer_links=("ctcf_enhancer", "sum"),
            promoter_ctcf_links=("bait_ctcf_count", lambda values: int((values > 0).sum())),
        )
        summary["culture_condition"] = state
        records.append(summary)
    return pd.concat(records, ignore_index=True)


def guide_summary(config: dict) -> pd.DataFrame:
    guides = pd.read_csv(SOURCE / "analyses/guide_dose_response/results/eligible_guide_rows.csv")
    pairs = pd.read_csv(SOURCE / "work/upstream/GWT_perturbseq_analysis_2025/metadata/DE_by_guide.correlation_results.csv", index_col=0)
    pair_columns = ["target", "culture_condition", "correlation_signif", "correlation_all", "n_signif_union", "frac_sign_agreement_signif_union", "n_signif_ontarget"]
    pairs = pairs[pair_columns].drop_duplicates(["target", "culture_condition"])
    summary = guides.groupby(["target", "culture_condition"], as_index=False).agg(
        guide_count=("guide_id", "nunique"),
        significant_cis_guides=("signif_knockdown", "sum"),
        mean_knockdown_fraction=("knockdown_fraction", "mean"),
        knockdown_range=("knockdown_fraction", lambda values: values.max() - values.min()),
        minimum_guide_cells=("guide_n", "min"),
        guide_downstream_count_difference=("downstream_genes", lambda values: values.max() - values.min()),
    )
    summary = summary.merge(pairs, on=["target", "culture_condition"], how="left", validate="one_to_one")
    guide_config = config["guide_concordance"]
    summary["guide_concordant"] = (
        summary["significant_cis_guides"].ge(int(guide_config["minimum_guides"]))
        & summary["n_signif_union"].ge(int(guide_config["minimum_union_genes"]))
        & summary["correlation_all"].ge(float(guide_config["concordant_correlation_threshold"]))
        & summary["frac_sign_agreement_signif_union"].ge(float(guide_config["concordant_sign_threshold"]))
    )
    return summary.rename(columns={"target": "target_gene"})


def build_edge_matrix(
    rows: pd.DataFrame,
    genes: pd.DataFrame,
    response_matrix: sparse.csr_matrix,
    curated: pd.DataFrame,
    motifs: dict[str, set[str]],
    pchic: pd.DataFrame,
    gene_table: pd.DataFrame,
    guide: pd.DataFrame,
    k562: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curated_primary = curated[curated["primary_curated"]].copy()
    curated_lookup = curated_primary.set_index(["source_gene", "response_gene"]).to_dict("index")
    curated_out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for edge in curated_primary.itertuples(index=False):
        curated_out[edge.source_gene].append((edge.response_gene, int(edge.curated_direction)))
    eligible_regulators = set(curated_primary["source_gene"]) | set(motifs)
    pchic_lookup = pchic.set_index(["gene", "culture_condition"]).to_dict("index")
    guide_lookup = guide.set_index(["target_gene", "culture_condition"]).to_dict("index")
    k562_lookup = k562.set_index(["target_gene", "culture_condition"]).to_dict("index")
    gene_names = genes["gene_name"].astype(str).str.upper().to_numpy()
    gene_to_index = {gene: index for index, gene in enumerate(gene_names)}
    promoter_ctcf = gene_table.set_index("gene_name")["ctcf_peak_count"].to_dict()
    edge_records = []
    target_records = []
    for response_row, row in rows.reset_index(drop=True).iterrows():
        source = str(row["target_contrast_gene_name"]).upper()
        state = str(row["culture_condition"])
        eligible = source in eligible_regulators
        nonzero = response_matrix.getrow(response_row)
        response_indices = nonzero.indices
        response_values = nonzero.data
        response_effect = {gene_names[index]: float(effect) for index, effect in zip(response_indices, response_values, strict=True)}
        guide_row = guide_lookup.get((source, state), {})
        guide_concordant = bool(guide_row.get("guide_concordant", False))
        k562_row = k562_lookup.get((source, state), {})
        independent_concordant = bool(
            np.isfinite(k562_row.get("correlation_delta", np.nan))
            and k562_row.get("correlation_delta", np.nan) > 0
            and k562_row.get("logfc_pearson_pval", 1.0) < 0.05
        )
        if not eligible:
            target_records.append({
                "response_row": response_row,
                "index": row["index"],
                "target_gene": source,
                "target_contrast": row["target_contrast"],
                "culture_condition": state,
                "eligible_regulator": False,
                "eligibility_status": "ineligible_no_compatible_regulatory_resource",
                "significant_response_edges": int(len(response_indices)),
            })
            continue
        higher_paths = {}
        path_candidates = defaultdict(list)
        for intermediate, intermediate_effect in response_effect.items():
            if intermediate == source or intermediate not in curated_out:
                continue
            for distal, direction in curated_out[intermediate]:
                if distal not in response_effect or distal == source or direction == 0:
                    continue
                agreement = int(np.sign(response_effect[distal]) == np.sign(intermediate_effect) * direction)
                path_candidates[distal].append((intermediate, direction, agreement))
                rank = (abs(intermediate_effect), intermediate)
                current = higher_paths.get(distal)
                if current is None or rank > current[0]:
                    higher_paths[distal] = (rank, intermediate, direction, bool(agreement))
        motif_targets = motifs.get(source, set())
        pchic_state = pchic_lookup
        tier_counts = defaultdict(int)
        class_counts = defaultdict(int)
        signed_agreements = []
        for gene_index, effect in zip(response_indices, response_values, strict=True):
            response_gene = gene_names[gene_index]
            curated_edge = curated_lookup.get((source, response_gene))
            curated_supported = curated_edge is not None
            curated_direction = int(curated_edge["curated_direction"]) if curated_supported else 0
            direction_conflict = bool(curated_edge["direction_conflict"]) if curated_supported else False
            cis_sign = int(np.sign(row["ontarget_effect_size"]))
            expected_effect_sign = cis_sign * curated_direction if curated_direction else 0
            direction_agreement = bool(curated_direction and np.sign(effect) == expected_effect_sign)
            direction_contradiction = bool(curated_direction and not direction_agreement) or direction_conflict
            motif_supported = response_gene in motif_targets
            contact = pchic_state.get((response_gene, state), {})
            physical_active = int(contact.get("active_links", 0)) > 0
            occupancy_compatible = source == "CTCF" and state == "Rest"
            promoter_bound = occupancy_compatible and int(promoter_ctcf.get(response_gene, 0)) > 0
            enhancer_linked = occupancy_compatible and int(contact.get("ctcf_enhancer_links", 0)) > 0
            higher = higher_paths.get(response_gene)
            higher_order = higher is not None and not curated_supported
            if enhancer_linked:
                edge_class = "enhancer-linked"
            elif promoter_bound:
                edge_class = "promoter-bound"
            elif motif_supported:
                edge_class = "motif-supported"
            elif curated_supported:
                edge_class = "one-step TF cascade"
            elif higher_order:
                edge_class = "higher-order cascade"
            else:
                edge_class = "unsupported indirect"
            independent_types = int(curated_supported) + int(motif_supported) + int(promoter_bound or enhancer_linked) + int(physical_active)
            tier_1 = enhancer_linked and physical_active and guide_concordant and not direction_contradiction
            if tier_1:
                tier = "Tier 1"
            elif curated_supported and curated_direction and direction_agreement and independent_types >= 3:
                tier = "Tier 2"
            elif independent_types >= 2 and not direction_contradiction:
                tier = "Tier 3"
            else:
                tier = "Tier 4"
            if curated_direction:
                direction_status = "concordant" if direction_agreement else "contradictory"
                signed_agreements.append(float(direction_agreement))
            elif direction_conflict:
                direction_status = "conflicting_curated_direction"
            else:
                direction_status = "unavailable"
            higher_intermediate = higher[1] if higher else ""
            higher_direction = higher[2] if higher else 0
            higher_agreement = higher[3] if higher else False
            edge_records.append({
                "response_row": response_row,
                "gene_column": int(gene_index),
                "index": row["index"],
                "target_gene": source,
                "target_contrast": row["target_contrast"],
                "culture_condition": state,
                "response_gene": response_gene,
                "crispr_effect": float(effect),
                "crispr_sign": int(np.sign(effect)),
                "curated_edge": curated_supported,
                "curated_direction": curated_direction,
                "expected_crispr_sign": expected_effect_sign,
                "cis_sign": cis_sign,
                "direction_status": direction_status,
                "motif_supported": motif_supported,
                "physical_active_link": physical_active,
                "occupancy_compatible": occupancy_compatible,
                "promoter_bound": promoter_bound,
                "enhancer_linked": enhancer_linked,
                "higher_order_intermediate": higher_intermediate,
                "higher_order_direction": higher_direction,
                "higher_order_sign_agreement": higher_agreement,
                "candidate_path_count": len(path_candidates.get(response_gene, [])),
                "candidate_agreeing_paths": sum(value[2] for value in path_candidates.get(response_gene, [])),
                "candidate_conflicting_paths": sum(1 - value[2] for value in path_candidates.get(response_gene, [])),
                "candidate_intermediates": ";".join(sorted(value[0] for value in path_candidates.get(response_gene, []))),
                "guide_concordant": guide_concordant,
                "k562_vector_concordant": independent_concordant,
                "independent_evidence_types": independent_types,
                "validation_evidence_types": int(guide_concordant) + int(independent_concordant),
                "edge_class": edge_class,
                "evidence_tier": tier,
                "directional_contradiction": direction_contradiction,
            })
            tier_counts[tier] += 1
            class_counts[edge_class] += 1
        target_records.append({
            "response_row": response_row,
            "index": row["index"],
            "target_gene": source,
            "target_contrast": row["target_contrast"],
            "culture_condition": state,
            "eligible_regulator": True,
            "eligibility_status": "eligible" if len(response_indices) else "eligible_null_response",
            "significant_response_edges": int(len(response_indices)),
            "tier_1_edges": tier_counts["Tier 1"],
            "tier_2_edges": tier_counts["Tier 2"],
            "tier_3_edges": tier_counts["Tier 3"],
            "tier_4_edges": tier_counts["Tier 4"],
            "promoter_bound_edges": class_counts["promoter-bound"],
            "enhancer_linked_edges": class_counts["enhancer-linked"],
            "motif_supported_edges": class_counts["motif-supported"],
            "one_step_edges": class_counts["one-step TF cascade"],
            "higher_order_edges": class_counts["higher-order cascade"],
            "unsupported_indirect_edges": class_counts["unsupported indirect"],
            "signed_direction_agreement": float(np.mean(signed_agreements)) if signed_agreements else np.nan,
            "guide_concordant": guide_concordant,
            "k562_vector_concordant": independent_concordant,
        })
    edges = pd.DataFrame(edge_records)
    targets = pd.DataFrame(target_records)
    denominator = targets["significant_response_edges"].replace(0, np.nan)
    for column in ["tier_1_edges", "tier_2_edges", "tier_3_edges", "tier_4_edges", "promoter_bound_edges", "enhancer_linked_edges", "motif_supported_edges", "one_step_edges", "higher_order_edges", "unsupported_indirect_edges"]:
        if column not in targets:
            targets[column] = 0
        targets[column] = targets[column].fillna(0).astype(int)
        targets[column.replace("_edges", "_fraction")] = targets[column] / denominator
    targets["direct_supported_edges"] = targets[["promoter_bound_edges", "enhancer_linked_edges", "motif_supported_edges", "one_step_edges"]].sum(axis=1)
    targets["direct_support_fraction"] = targets["direct_supported_edges"] / denominator
    targets["occupied_fraction"] = (targets["promoter_bound_edges"] + targets["enhancer_linked_edges"]) / denominator
    targets["supported_path_depth"] = (targets["direct_supported_edges"] + 2 * targets["higher_order_edges"]) / (targets["direct_supported_edges"] + targets["higher_order_edges"]).replace(0, np.nan)
    return edges, targets


def infer_tf_activity(
    rows: pd.DataFrame,
    response_matrix: sparse.csr_matrix,
    genes: pd.DataFrame,
    curated: pd.DataFrame,
    eligible_rows: np.ndarray,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum = int(config["eligibility"]["activity_minimum_regulon_targets"])
    gene_names = genes["gene_name"].astype(str).str.upper().to_numpy()
    gene_index = {gene: index for index, gene in enumerate(gene_names)}

    def score_network(network: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        network = network[network["curated_direction"].ne(0)].copy()
        counts = network.groupby("source_gene")["response_gene"].nunique()
        regulators = sorted(counts[counts.ge(minimum)].index)
        regulator_index = {regulator: index for index, regulator in enumerate(regulators)}
        r, c, values = [], [], []
        for edge in network[network["source_gene"].isin(regulators)].itertuples(index=False):
            if edge.response_gene in gene_index:
                r.append(gene_index[edge.response_gene])
                c.append(regulator_index[edge.source_gene])
                values.append(float(edge.curated_direction))
        weights = sparse.csr_matrix((values, (r, c)), shape=(len(gene_names), len(regulators)))
        raw = (response_matrix[eligible_rows] @ weights).toarray()
        target_count = np.asarray(weights.power(2).sum(axis=0)).ravel()
        weighted_mean = raw / np.maximum(target_count, 1)[None, :]
        n = response_matrix.shape[1]
        response_sum = np.asarray(response_matrix[eligible_rows].sum(axis=1)).ravel()
        response_ss = np.asarray(response_matrix[eligible_rows].power(2).sum(axis=1)).ravel() - response_sum ** 2 / n
        weight_sum = np.asarray(weights.sum(axis=0)).ravel()
        weight_ss = target_count - weight_sum ** 2 / n
        covariance = raw - response_sum[:, None] * weight_sum[None, :] / n
        correlation = covariance / np.sqrt(np.maximum(response_ss[:, None] * weight_ss[None, :], 1e-24))
        correlation = np.clip(correlation, -1 + 1e-12, 1 - 1e-12)
        ulm = correlation * np.sqrt((n - 2) / np.maximum(1 - correlation ** 2, 1e-24))
        return regulators, weighted_mean, ulm, target_count

    primary = curated[curated["dorothea_a"]]
    regulators, weighted_mean, ulm, target_count = score_network(primary)
    standard = []
    for matrix in [weighted_mean, ulm]:
        mean = np.nanmean(matrix, axis=0)
        sd = np.nanstd(matrix, axis=0)
        standard.append((matrix - mean) / np.where(sd > 0, sd, 1.0))
    consensus = (standard[0] + standard[1]) / 2
    top_n = int(config["inference"]["activity_top_per_target_state"])
    top_records = []
    source_records = []
    regulator_index = {regulator: index for index, regulator in enumerate(regulators)}
    for local_index, response_row in enumerate(eligible_rows):
        order = np.argsort(np.abs(consensus[local_index]))[::-1][:top_n]
        for rank, column in enumerate(order, start=1):
            top_records.append({
                "response_row": int(response_row),
                "index": rows.iloc[response_row]["index"],
                "target_gene": str(rows.iloc[response_row]["target_contrast_gene_name"]).upper(),
                "culture_condition": rows.iloc[response_row]["culture_condition"],
                "inferred_tf": regulators[column],
                "weighted_mean_score": weighted_mean[local_index, column],
                "ulm_score": ulm[local_index, column],
                "consensus_score": consensus[local_index, column],
                "absolute_rank": rank,
                "regulon_targets": int(target_count[column]),
            })
        source = str(rows.iloc[response_row]["target_contrast_gene_name"]).upper()
        column = regulator_index.get(source)
        source_records.append({
            "response_row": int(response_row),
            "source_tf_activity": consensus[local_index, column] if column is not None else np.nan,
            "source_tf_weighted_mean": weighted_mean[local_index, column] if column is not None else np.nan,
            "source_tf_activity_status": "estimable" if column is not None else "unavailable_no_high_confidence_regulon",
        })
    broad = curated[curated["dorothea_any"] & (curated["dorothea_a"] | curated["dorothea_b"])]
    broad_regulators, broad_wmean, broad_ulm, broad_count = score_network(broad)
    sensitivity = pd.DataFrame({
        "analysis": ["DoRothEA A primary", "DoRothEA A-B sensitivity"],
        "regulators": [len(regulators), len(broad_regulators)],
        "signed_edges": [int(primary["curated_direction"].ne(0).sum()), int(broad["curated_direction"].ne(0).sum())],
        "median_regulon_targets": [float(np.median(target_count)) if len(target_count) else np.nan, float(np.median(broad_count)) if len(broad_count) else np.nan],
        "score_rows": [len(weighted_mean), len(broad_wmean)],
        "methods": ["weighted_mean;univariate_linear_model"] * 2,
    })
    return pd.DataFrame(top_records), pd.DataFrame(source_records), sensitivity


def build_matched_backgrounds(
    edges: pd.DataFrame,
    rows: pd.DataFrame,
    gene_table: pd.DataFrame,
    curated: pd.DataFrame,
    motifs: dict[str, set[str]],
    config: dict,
) -> pd.DataFrame:
    baseline = rows[["target_contrast_gene_name", "culture_condition", "target_baseMean"]].copy()
    baseline["gene_name"] = baseline["target_contrast_gene_name"].astype(str).str.upper()
    baseline = baseline.groupby(["gene_name", "culture_condition"], as_index=False)["target_baseMean"].median()
    primary = curated[curated["primary_curated"]]
    in_degree = primary.groupby("response_gene").size()
    out_degree = primary.groupby("source_gene").size()
    genes = gene_table.copy()
    genes["curated_in_degree"] = genes["gene_name"].map(in_degree).fillna(0)
    genes["curated_out_degree"] = genes["gene_name"].map(out_degree).fillna(0)
    genes = genes.merge(baseline, on="gene_name", how="inner")
    genes = genes.dropna(subset=["chromosome", "tss", "gene_gc_percent", "gene_interval_length"])
    gene_lookup = gene_table.set_index("gene_name").to_dict("index")
    curated_out = defaultdict(set)
    for row in primary.itertuples(index=False):
        curated_out[row.source_gene].add(row.response_gene)
    controls_per_case = int(config["motif_background"]["controls_per_case"])
    records = []
    for (response_row, source, state), response_edges in edges.groupby(["response_row", "target_gene", "culture_condition"], sort=False):
        source_data = gene_lookup.get(source)
        if source_data is None or not np.isfinite(source_data.get("tss", np.nan)):
            continue
        pool = genes[genes["culture_condition"].eq(state)].copy()
        if pool.empty:
            continue
        pool["chromosome_class"] = np.where(pool["chromosome"].eq(source_data["chromosome"]), "same", "different")
        pool["regulator_target_distance"] = np.where(
            pool["chromosome_class"].eq("same"),
            np.abs(pool["tss"] - source_data["tss"]),
            1e9,
        )
        pool["baseline_expression"] = np.log1p(pool["target_baseMean"].clip(lower=0))
        pool["atac_peak_count"] = np.log1p(pool["atac_peak_count"])
        pool["h3k27ac_peak_count"] = np.log1p(pool["h3k27ac_peak_count"])
        pool["gene_interval_length"] = np.log1p(pool["gene_interval_length"].clip(lower=1))
        pool["regulator_target_distance"] = np.log1p(pool["regulator_target_distance"])
        pool["curated_in_degree"] = np.log1p(pool["curated_in_degree"])
        pool["curated_out_degree"] = np.log1p(pool["curated_out_degree"])
        covariates = [
            "baseline_expression",
            "atac_peak_count",
            "h3k27ac_peak_count",
            "gene_gc_percent",
            "gene_interval_length",
            "regulator_target_distance",
            "curated_in_degree",
            "curated_out_degree",
        ]
        response_genes = set(response_edges["response_gene"])
        motif_targets = motifs.get(source, set())
        curated_targets = curated_out.get(source, set())
        for chromosome_class, cases in response_edges.groupby(response_edges["response_gene"].map(lambda gene: "same" if gene_lookup.get(gene, {}).get("chromosome") == source_data["chromosome"] else "different")):
            candidates = pool[pool["chromosome_class"].eq(chromosome_class) & ~pool["gene_name"].isin(response_genes)].copy()
            case_data = pool[pool["gene_name"].isin(cases["response_gene"]) & pool["chromosome_class"].eq(chromosome_class)].drop_duplicates("gene_name")
            if len(candidates) < controls_per_case or case_data.empty:
                continue
            combined = pd.concat([candidates[covariates], case_data[covariates]], ignore_index=True)
            median = combined.median()
            scale = combined.fillna(median).std(ddof=0).replace(0, 1)
            candidate_values = ((candidates[covariates].fillna(median) - median) / scale).to_numpy(dtype=float)
            case_values = ((case_data[covariates].fillna(median) - median) / scale).to_numpy(dtype=float)
            neighbors = NearestNeighbors(n_neighbors=min(max(controls_per_case * 3, 20), len(candidates)), algorithm="auto").fit(candidate_values)
            distances, indices = neighbors.kneighbors(case_values)
            for case_position, case_gene in enumerate(case_data["gene_name"]):
                if distances[case_position, controls_per_case - 1] > float(config["motif_background"]["standardized_distance_caliper"]):
                    continue
                match_id = f"{response_row}:{case_gene}"
                case_motif = case_gene in motif_targets
                case_curated = case_gene in curated_targets
                records.append({
                    "match_id": match_id,
                    "response_row": int(response_row),
                    "target_gene": source,
                    "culture_condition": state,
                    "case_gene": case_gene,
                    "matched_gene": case_gene,
                    "role": "case",
                    "slot": 0,
                    "chromosome_class": chromosome_class,
                    "motif_supported": case_motif,
                    "curated_supported": case_curated,
                    "match_distance": 0.0,
                })
                chosen = indices[case_position, :controls_per_case]
                for slot, candidate_index in enumerate(chosen, start=1):
                    control = candidates.iloc[int(candidate_index)]
                    records.append({
                        "match_id": match_id,
                        "response_row": int(response_row),
                        "target_gene": source,
                        "culture_condition": state,
                        "case_gene": case_gene,
                        "matched_gene": control["gene_name"],
                        "role": "control",
                        "slot": slot,
                        "chromosome_class": chromosome_class,
                        "motif_supported": control["gene_name"] in motif_targets,
                        "curated_supported": control["gene_name"] in curated_targets,
                        "match_distance": float(distances[case_position, slot - 1]),
                    })
    return pd.DataFrame(records)


def run_matched_permutations(matched: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(config["seed"]) + 201)
    replicates = int(config["inference"]["matched_permutation_replicates"])
    records = []
    summaries = []
    for state in STATE_ORDER:
        block = matched[matched["culture_condition"].eq(state)]
        for evidence in ["motif_supported", "curated_supported"]:
            pivot = block.pivot(index="match_id", columns="slot", values=evidence).dropna().astype(float)
            expected_columns = list(range(int(config["motif_background"]["controls_per_case"]) + 1))
            pivot = pivot.reindex(columns=expected_columns).dropna()
            if pivot.empty:
                summaries.append({"culture_condition": state, "evidence": evidence, "status": "unavailable", "matched_cases": 0})
                continue
            values = pivot.to_numpy(dtype=float)
            observed = float(np.mean(values[:, 0] - values[:, 1:].mean(axis=1)))
            total = values.sum(axis=1)
            null = np.empty(replicates, dtype=float)
            for replicate in range(replicates):
                chosen = rng.integers(0, values.shape[1], size=len(values))
                pseudo = values[np.arange(len(values)), chosen]
                null[replicate] = np.mean(pseudo - (total - pseudo) / (values.shape[1] - 1))
                records.append({"culture_condition": state, "evidence": evidence, "replicate": replicate + 1, "null_delta": null[replicate], "observed_delta": observed})
            p_value = (1 + int(np.sum(null >= observed - 1e-12))) / (replicates + 1)
            summaries.append({
                "culture_condition": state,
                "evidence": evidence,
                "status": "estimable",
                "matched_cases": int(len(values)),
                "observed_delta": observed,
                "null_mean": float(null.mean()),
                "null_ci_low": float(np.quantile(null, 0.025)),
                "null_ci_high": float(np.quantile(null, 0.975)),
                "p_value": p_value,
            })
    summary = pd.DataFrame(summaries)
    summary["q_value"] = bh(summary.get("p_value", pd.Series(dtype=float)).to_numpy())
    return pd.DataFrame(records), summary


def degree_preserving_null(edges: pd.DataFrame, curated: pd.DataFrame, config: dict) -> pd.DataFrame:
    response_pairs = set(zip(edges["target_gene"], edges["response_gene"], strict=True))
    eligible_sources = set(edges["target_gene"])
    graph = curated[curated["primary_curated"] & curated["source_gene"].isin(eligible_sources)]
    original = list(dict.fromkeys(zip(graph["source_gene"], graph["response_gene"], strict=True)))
    edge_set = set(original)
    source_degree = Counter(a for a, _ in original)
    target_degree = Counter(b for _, b in original)
    observed = len(edge_set & response_pairs) / max(len(edge_set), 1)
    rng = np.random.default_rng(int(config["seed"]) + 311)
    replicates = int(config["inference"]["degree_rewiring_replicates"])
    records = []
    for replicate in range(replicates):
        current = list(original)
        current_set = set(current)
        successful = 0
        attempts = 0
        target_swaps = len(current)
        while successful < target_swaps and attempts < target_swaps * 12:
            first, second = rng.integers(0, len(current), size=2)
            attempts += 1
            if first == second:
                continue
            a, b = current[first]
            c, d = current[second]
            proposed_first, proposed_second = (a, d), (c, b)
            if a == d or c == b or proposed_first in current_set or proposed_second in current_set:
                continue
            current_set.remove((a, b))
            current_set.remove((c, d))
            current[first], current[second] = proposed_first, proposed_second
            current_set.add(proposed_first)
            current_set.add(proposed_second)
            successful += 1
        overlap = len(current_set & response_pairs) / max(len(current_set), 1)
        preserved = Counter(a for a, _ in current_set) == source_degree and Counter(b for _, b in current_set) == target_degree
        if not preserved:
            raise ValueError("Rewiring changed regulatory in-degree or out-degree")
        records.append({
            "replicate": replicate + 1,
            "rewired_overlap_fraction": overlap,
            "observed_overlap_fraction": observed,
            "edges": len(current_set),
            "successful_swaps": successful,
            "degree_preserved": preserved,
        })
    return pd.DataFrame(records)


def bootstrap_statistic(table: pd.DataFrame, group: str, statistic, replicates: int, seed: int) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    table = table.reset_index(drop=True)
    grouped = list(table.groupby(group, sort=False).indices.values())
    estimates = np.full(replicates, np.nan)
    for replicate in range(replicates):
        sample = rng.integers(0, len(grouped), size=len(grouped))
        resampled = table.iloc[np.concatenate([grouped[index] for index in sample])].reset_index(drop=True)
        estimates[replicate] = statistic(resampled)
    valid = estimates[np.isfinite(estimates)]
    if not len(valid):
        return np.nan, np.nan, estimates
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975)), estimates


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 4 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def run_predictive_validation(targets: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = ["cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "log1p_n_downstream"]
    regulatory = ["direct_support_fraction", "higher_order_fraction", "tier_2_fraction", "tier_3_fraction", "signed_direction_agreement", "source_tf_weighted_mean"]
    datasets = {
        "guide_concordance": ("correlation_all", targets["n_signif_union"].ge(int(config["guide_concordance"]["minimum_union_genes"]))),
        "k562_replication": ("correlation_delta", targets["correlation_delta"].notna()),
    }
    prediction_records = []
    metric_records = []
    for dataset, (outcome, eligible) in datasets.items():
        table = targets[eligible & targets[outcome].notna() & targets["eligible_regulator"]].copy()
        table = pd.concat([table, pd.get_dummies(table["culture_condition"], prefix="state", dtype=float)], axis=1)
        state_columns = [column for column in table if column.startswith("state_")]
        table[baseline + regulatory] = table[baseline + regulatory].replace([np.inf, -np.inf], np.nan)
        table["fold"] = table["target_gene"].map(lambda value: stable_fold(str(value), int(config["inference"]["target_holdout_folds"]), int(config["seed"])))
        for model, columns in {"covariates": baseline + state_columns, "regulatory_support": baseline + regulatory + state_columns}.items():
            started = time.perf_counter()
            observed_all, predicted_all = [], []
            rows = []
            for fold in sorted(table["fold"].unique()):
                train = table[table["fold"].ne(fold)]
                test = table[table["fold"].eq(fold)]
                estimator = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
                estimator.fit(train[columns], train[outcome])
                train_prediction = estimator.predict(train[columns])
                prediction = estimator.predict(test[columns])
                residual = train[outcome].to_numpy(dtype=float) - train_prediction
                quantiles = {level: np.quantile(np.abs(residual), level) for level in [0.80, 0.95]}
                observed_all.extend(test[outcome].to_numpy(dtype=float))
                predicted_all.extend(prediction)
                for position, (_, row) in enumerate(test.iterrows()):
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "index": row["index"],
                        "target_gene": row["target_gene"],
                        "culture_condition": row["culture_condition"],
                        "fold": int(fold),
                        "observed": row[outcome],
                        "predicted": prediction[position],
                        "interval_method": "training_residual_quantile_not_conformal",
                        "interval_80_low": prediction[position] - quantiles[0.80],
                        "interval_80_high": prediction[position] + quantiles[0.80],
                        "interval_95_low": prediction[position] - quantiles[0.95],
                        "interval_95_high": prediction[position] + quantiles[0.95],
                    })
            prediction_table = pd.DataFrame(rows)
            prediction_records.append(prediction_table)
            observed = np.asarray(observed_all)
            predicted = np.asarray(predicted_all)
            slope, intercept = np.polyfit(predicted, observed, 1) if np.std(predicted) > 0 else (np.nan, np.nan)
            metric_records.append({
                "dataset": dataset,
                "model": model,
                "rows": len(prediction_table),
                "targets": prediction_table["target_gene"].nunique(),
                "r2": r2_score(observed, predicted),
                "rmse": mean_squared_error(observed, predicted) ** 0.5,
                "spearman_r": safe_spearman(pd.Series(observed), pd.Series(predicted))[0],
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "coverage_80": prediction_table["observed"].between(prediction_table["interval_80_low"], prediction_table["interval_80_high"]).mean(),
                "coverage_95": prediction_table["observed"].between(prediction_table["interval_95_low"], prediction_table["interval_95_high"]).mean(),
                "elapsed_seconds": time.perf_counter() - started,
                "features": ";".join(columns),
            })
    return pd.concat(prediction_records, ignore_index=True), pd.DataFrame(metric_records)


def difference_statistic(table: pd.DataFrame, group_column: str, positive: str | bool, negative: str | bool, outcome: str) -> float:
    first = table.loc[table[group_column].eq(positive), outcome].dropna()
    second = table.loc[table[group_column].eq(negative), outcome].dropna()
    return float(first.mean() - second.mean()) if len(first) and len(second) else np.nan


def compile_hypotheses(
    targets: pd.DataFrame,
    edges: pd.DataFrame,
    rewiring: pd.DataFrame,
    natural: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    replicates = int(config["inference"]["bootstrap_replicates"])
    seed = int(config["seed"])
    records = []

    def add(name: str, family: str, estimate: float, ci: tuple[float, float], p_value: float, n: int, expected: str, note: str = "") -> None:
        records.append({"hypothesis": name, "family": family, "estimate": estimate, "ci_low": ci[0], "ci_high": ci[1], "p_value": p_value, "n": n, "expected_direction": expected, "note": note})

    def clustered(table: pd.DataFrame, statistic, offset: int) -> tuple[float, tuple[float, float], float, int]:
        if table["target_gene"].nunique() < 2:
            return np.nan, (np.nan, np.nan), np.nan, table["target_gene"].nunique()
        estimate = statistic(table)
        low, high, draws = bootstrap_statistic(table, "target_gene", statistic, replicates, seed + offset)
        valid = draws[np.isfinite(draws)]
        p_value = (1 + np.sum(np.abs(valid - estimate) >= abs(estimate))) / (len(valid) + 1)
        return estimate, (low, high), float(p_value), table["target_gene"].nunique()

    eligible = targets[targets["eligible_regulator"] & targets["significant_response_edges"].gt(0)].copy()
    statistic = lambda data: safe_spearman(data["transfer_residual"], data["direct_support_fraction"])[0]
    estimate, ci, p_value, n = clustered(eligible, statistic, 1)
    add("Regulatory support increases with transfer gain", "transfer_gain", estimate, ci, p_value, n, "positive", "target-cluster bootstrap; motif or curated support is not occupancy")

    tails = eligible[eligible["transfer_class"].isin(["amplified", "buffered"])]
    estimate, ci, p_value, n = clustered(tails, lambda data: difference_statistic(data, "transfer_class", "amplified", "buffered", "direct_support_fraction"), 2)
    add("Amplified responses have stronger regulatory support than buffered responses", "amplification_buffering", estimate, ci, p_value, n, "positive", "secondary support proxy; direct occupancy comparison unavailable")

    estimate, ci, p_value, n = clustered(eligible.dropna(subset=["mean_rerouting_js"]), lambda data: safe_spearman(data["mean_rerouting_js"], data["higher_order_fraction"])[0], 3)
    add("Rerouting increases with higher-order cascade support", "rerouting", estimate, ci, p_value, n, "positive", "target-cluster bootstrap")

    timing = eligible.pivot_table(index="target_gene", columns="culture_condition", values=["direct_support_fraction", "supported_path_depth"], aggfunc="mean").dropna()
    if len(timing):
        timing_rows = timing.reset_index()
        direct_estimate = float(timing["direct_support_fraction"]["Rest"].mean() - timing["direct_support_fraction"][["Stim8hr", "Stim48hr"]].mean(axis=1).mean())
        depth_estimate = float(timing["supported_path_depth"][["Stim8hr", "Stim48hr"]].mean(axis=1).mean() - timing["supported_path_depth"]["Rest"].mean())
        direct_diff = timing["direct_support_fraction"]["Rest"] - timing["direct_support_fraction"][["Stim8hr", "Stim48hr"]].mean(axis=1)
        depth_diff = timing["supported_path_depth"][["Stim8hr", "Stim48hr"]].mean(axis=1) - timing["supported_path_depth"]["Rest"]
        rng = np.random.default_rng(seed + 4)
        direct_boot = np.array([direct_diff.iloc[rng.integers(0, len(direct_diff), len(direct_diff))].mean() for _ in range(replicates)])
        depth_boot = np.array([depth_diff.iloc[rng.integers(0, len(depth_diff), len(depth_diff))].mean() for _ in range(replicates)])
        add("Resting responses contain more regulatory support", "timing_proxy", direct_estimate, tuple(np.quantile(direct_boot, [0.025, 0.975])), float(wilcoxon(direct_diff).pvalue), len(timing_rows), "positive", "matched targets; resting versus stimulated states, not temporal causation")
        add("Later responses have greater supported path depth", "timing", depth_estimate, tuple(np.quantile(depth_boot, [0.025, 0.975])), float(wilcoxon(depth_diff, alternative="greater").pvalue), len(timing_rows), "positive", "cross-sectional state comparison")
    add("Early responses are enriched for directly occupied edges", "direct_timing", np.nan, (np.nan, np.nan), np.nan, 0, "positive", "unavailable: no eligible perturbation has compatible TF occupancy")

    guide_rows = eligible[eligible["n_signif_union"].ge(int(config["guide_concordance"]["minimum_union_genes"]))]
    estimate, ci, p_value, n = clustered(guide_rows, lambda data: safe_spearman(data["direct_support_fraction"], data["correlation_all"])[0], 5)
    add("Regulatory support predicts cross-guide concordance", "guide_concordance", estimate, ci, p_value, n, "positive", "guide evidence excluded from predictor")

    external = eligible[eligible["correlation_delta"].notna()]
    estimate, ci, p_value, n = clustered(external, lambda data: safe_spearman(data["direct_support_fraction"], data["correlation_delta"])[0], 6)
    add("Regulatory support predicts K562 response-vector reproducibility", "external_replication", estimate, ci, p_value, n, "positive", "K562 evidence excluded from predictor")

    signed = eligible.dropna(subset=["signed_direction_agreement"])
    estimate, ci, p_value, n = clustered(signed, lambda data: float(data.signed_direction_agreement.mean() - 0.5), 7)
    add("Signed CRISPR effects agree with curated regulatory direction", "signed_propagation", estimate, ci, p_value, n, "positive", "target-cluster bootstrap")

    if len(natural):
        natural["supported_tier"] = natural["evidence_tier"].isin(["Tier 1", "Tier 2", "Tier 3"])
        estimate = difference_statistic(natural, "supported_tier", True, False, "direction_match")
        estimate, ci, p_value, n = clustered(natural, lambda data: difference_statistic(data, "supported_tier", True, False, "direction_match"), 8)
        first = natural.loc[natural["supported_tier"], "direction_match"]
        second = natural.loc[~natural["supported_tier"], "direction_match"]
        table = [[int(first.sum()), int(len(first) - first.sum())], [int(second.sum()), int(len(second) - second.sum())]]
        add("Regulatory support predicts natural-genetic direction concordance", "natural_genetics", estimate, ci, p_value, n, "positive", "target-cluster bootstrap")

    disease = eligible[eligible["disease_program_convergence"].notna()]
    estimate = difference_statistic(disease, "disease_program_convergence", True, False, "direct_support_fraction")
    estimate, ci, p_value, n = clustered(disease, lambda data: difference_statistic(data, "disease_program_convergence", True, False, "direct_support_fraction"), 9)
    first = disease.loc[disease["disease_program_convergence"].eq(True), "direct_support_fraction"].dropna()
    second = disease.loc[disease["disease_program_convergence"].eq(False), "direct_support_fraction"].dropna()
    add("Disease-convergent programs have stronger regulatory support", "disease_convergence", estimate, ci, p_value, n, "positive", "target-cluster bootstrap; broad programme membership is not causal evidence")

    observed = float(rewiring["observed_overlap_fraction"].iloc[0])
    null = rewiring["rewired_overlap_fraction"].to_numpy()
    estimate = observed - float(null.mean())
    p_value = (1 + int(np.sum(null >= observed))) / (len(null) + 1)
    add("Observed response edges exceed degree-preserving network nulls", "network_null", estimate, tuple(observed - np.quantile(null, [0.975, 0.025])), p_value, len(rewiring), "positive")
    records[-1]["note"] = "interval is the central 95% null-reference interval, not a sampling confidence interval; one successful swap per edge"

    result = pd.DataFrame(records)
    result["q_value"] = bh(result["p_value"].to_numpy())
    result["decision"] = "unresolved"
    positive = result["expected_direction"].eq("positive")
    result.loc[positive & result["ci_low"].gt(0) & result["q_value"].lt(float(config["inference"]["fdr"])), "decision"] = "passed"
    result.loc[positive & result["ci_high"].lt(0) & result["q_value"].lt(float(config["inference"]["fdr"])), "decision"] = "failed"
    result.loc[result["n"].lt(20), "decision"] = "underpowered"
    result.loc[result["note"].str.startswith("unavailable"), "decision"] = "unavailable"
    return result


def component_ablation(edges: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    edge_map = edges[["response_row", "motif_supported", "curated_edge", "promoter_bound", "enhancer_linked", "physical_active_link", "guide_concordant", "k562_vector_concordant"]].copy()
    class_map = targets.set_index("response_row")["transfer_class"].to_dict()
    records = []
    components = ["none", "motif", "curated", "occupancy", "physical_link", "guide_concordance", "external_replication"]
    for component in components:
        motif = edge_map["motif_supported"] & (component != "motif")
        curated = edge_map["curated_edge"] & (component != "curated")
        promoter = edge_map["promoter_bound"] & (component != "occupancy")
        enhancer = edge_map["enhancer_linked"] & (component not in ["occupancy", "physical_link"])
        direct = motif | curated | promoter | enhancer
        table = edge_map.assign(direct=direct).groupby("response_row", as_index=False).agg(edges=("direct", "size"), direct_edges=("direct", "sum"))
        table["direct_support_fraction"] = table["direct_edges"] / table["edges"]
        table["transfer_class"] = table["response_row"].map(class_map)
        estimate = difference_statistic(table, "transfer_class", "amplified", "buffered", "direct_support_fraction")
        records.append({
            "removed_component": component,
            "response_edges": int(table["edges"].sum()),
            "direct_edges": int(table["direct_edges"].sum()),
            "direct_fraction": float(table["direct_edges"].sum() / table["edges"].sum()),
            "amplified_minus_buffered_direct_fraction": estimate,
            "guide_requirement_removed": component == "guide_concordance",
            "external_requirement_removed": component == "external_replication",
        })
    return pd.DataFrame(records)


def threshold_sensitivity(
    links: pd.DataFrame,
    gene_table: pd.DataFrame,
    edges: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    records = []
    ctcf_edges = edges[edges["target_gene"].eq("CTCF") & edges["culture_condition"].eq("Rest")]
    ctcf_bed = read_bed(SOURCE / "work/regulatory_inputs/ENCFF858TLX.bed.gz")
    gene_lookup = gene_table.set_index("gene_name").to_dict("index")
    for threshold in config["chromatin"]["pchic_sensitivity_thresholds"]:
        for active_rule in config["robustness"]["alternative_active_chromatin_rules"]:
            support = pchic_support(links, config, float(threshold), active_rule)
            support_set = set(zip(support.loc[support["active_links"].gt(0), "gene"], support.loc[support["active_links"].gt(0), "culture_condition"], strict=True))
            for upstream in config["robustness"]["alternative_promoter_upstream_bp"]:
                promoter = set()
                for gene in ctcf_edges["response_gene"].unique():
                    values = gene_lookup.get(gene, {})
                    if not values or not np.isfinite(values.get("tss", np.nan)):
                        continue
                    downstream = int(config["chromatin"]["promoter_downstream_bp"])
                    if values["strand"] == 1:
                        start, end = values["tss"] - int(upstream), values["tss"] + downstream
                    else:
                        start, end = values["tss"] - downstream, values["tss"] + int(upstream)
                    if overlap_count(ctcf_bed, values["chromosome"], max(0, start), end) > 0:
                        promoter.add(gene)
                edge_keys = pd.MultiIndex.from_frame(edges[["response_gene", "culture_condition"]])
                physical = edge_keys.isin(pd.MultiIndex.from_tuples(sorted(support_set)))
                promoter_edge = edges["target_gene"].eq("CTCF") & edges["culture_condition"].eq("Rest") & edges["response_gene"].isin(promoter)
                records.append({
                    "pchic_threshold": float(threshold),
                    "active_chromatin_rule": active_rule,
                    "promoter_upstream_bp": int(upstream),
                    "physical_active_edges": int(np.sum(physical)),
                    "ctcf_promoter_bound_edges": int(promoter_edge.sum()),
                    "response_edges": len(edges),
                    "physical_active_fraction": float(np.mean(physical)),
                })
    return pd.DataFrame(records)


def leave_one_influence(targets: pd.DataFrame, ablation: pd.DataFrame) -> pd.DataFrame:
    table = targets[targets["eligible_regulator"] & targets["significant_response_edges"].gt(0) & targets["transfer_class"].isin(["amplified", "buffered"])].copy()
    base = difference_statistic(table, "transfer_class", "amplified", "buffered", "direct_support_fraction")
    records = [{"unit_type": "none", "unit": "none", "rows_remaining": len(table), "estimate": base, "change_from_complete": 0.0}]
    for unit_type, column in [("factor", "target_gene"), ("state", "culture_condition"), ("target_state", "index")]:
        for unit in sorted(table[column].dropna().astype(str).unique()):
            subset = table[table[column].astype(str).ne(unit)]
            estimate = difference_statistic(subset, "transfer_class", "amplified", "buffered", "direct_support_fraction")
            records.append({"unit_type": unit_type, "unit": unit, "rows_remaining": len(subset), "estimate": estimate, "change_from_complete": estimate - base})
    complete = ablation.loc[ablation["removed_component"].eq("none"), "amplified_minus_buffered_direct_fraction"].iloc[0]
    for row in ablation[ablation["removed_component"].ne("none")].itertuples(index=False):
        records.append({"unit_type": "dataset_component", "unit": row.removed_component, "rows_remaining": len(table), "estimate": row.amplified_minus_buffered_direct_fraction, "change_from_complete": row.amplified_minus_buffered_direct_fraction - complete})
    return pd.DataFrame(records)


def guide_tier_summary(targets: pd.DataFrame, config: dict) -> pd.DataFrame:
    table = targets[targets["eligible_regulator"] & targets["n_signif_union"].notna()].copy()
    table["best_evidence_tier"] = np.select(
        [table["tier_1_edges"].gt(0), table["tier_2_edges"].gt(0), table["tier_3_edges"].gt(0)],
        ["Tier 1", "Tier 2", "Tier 3"],
        default="Tier 4",
    )
    records = []
    for (state, tier), block in table.groupby(["culture_condition", "best_evidence_tier"]):
        values = block["correlation_all"].dropna().to_numpy()
        rng = np.random.default_rng(int(config["seed"]) + stable_fold(f"{state}:{tier}", 10000, 17))
        draws = np.median(values[rng.integers(0, len(values), size=(int(config["inference"]["bootstrap_replicates"]), len(values)))], axis=1) if len(values) else np.array([np.nan])
        records.append({
            "culture_condition": state,
            "evidence_tier": tier,
            "target_states": len(block),
            "targets": block["target_gene"].nunique(),
            "median_full_vector_pearson": block["correlation_all"].median(),
            "median_pearson_ci_low": float(np.nanquantile(draws, 0.025)),
            "median_pearson_ci_high": float(np.nanquantile(draws, 0.975)),
            "median_significant_union_pearson": block["correlation_signif"].median(),
            "median_sign_agreement": block["frac_sign_agreement_signif_union"].median(),
            "median_significant_union_genes": block["n_signif_union"].median(),
            "cis_both_guides_significant_fraction": block["significant_cis_guides"].ge(2).mean(),
            "concordant_fraction": block["guide_concordant"].mean(),
            "power_status": "adequate" if len(block) >= int(config["eligibility"]["guide_minimum_pairs"]) else "underpowered",
            "tier_definition": "molecular_only_excludes_guide_and_external_outcomes",
        })
    return pd.DataFrame(records)


def central_verification(hypotheses: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "transfer_gain": "Regulatory support increases with transfer gain",
        "amplification": "Amplified responses have stronger regulatory support than buffered responses",
        "buffering": "Amplified responses have stronger regulatory support than buffered responses",
        "rerouting": "Rerouting increases with higher-order cascade support",
        "context_switching": "Early responses are enriched for directly occupied edges",
        "cross_system_conservation": "Regulatory support predicts K562 response-vector reproducibility",
        "natural_genetic_concordance": "Regulatory support predicts natural-genetic direction concordance",
        "disease_convergence": "Disease-convergent programs have stronger regulatory support",
        "external_benchmarking": None,
        "guide_level_dose_response": "Regulatory support predicts cross-guide concordance",
    }
    lookup = hypotheses.set_index("hypothesis").to_dict("index")
    records = []
    eligible = targets[targets["eligible_regulator"]]
    for result, hypothesis in mapping.items():
        if result == "external_benchmarking":
            records.append({
                "central_result": result,
                "eligible_target_states": int(len(eligible)),
                "hypothesis": "Signed RPE1 verification",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "q_value": np.nan,
                "decision": "unavailable",
                "evidential_boundary": "RPE1 benchmark retained a scalar target endpoint, not compatible signed target-gene vectors",
            })
            continue
        row = lookup.get(hypothesis, {})
        records.append({
            "central_result": result,
            "eligible_target_states": int(len(eligible)),
            "hypothesis": hypothesis,
            "estimate": row.get("estimate", np.nan),
            "ci_low": row.get("ci_low", np.nan),
            "ci_high": row.get("ci_high", np.nan),
            "q_value": row.get("q_value", np.nan),
            "decision": row.get("decision", "unresolved"),
            "evidential_boundary": "shared amplification-buffering contrast" if result == "buffering" else "",
        })
    return pd.DataFrame(records)


def locus_verification(targets: pd.DataFrame) -> pd.DataFrame:
    loci = pd.read_csv(SOURCE / "analyses/loci/results/locus_summary.csv")
    gates = pd.read_csv(SOURCE / "analyses/causal_triangulation/results/evidence_gates.csv")
    grades = pd.read_csv(SOURCE / "analyses/causal_triangulation/results/locus_grades.csv")
    records = []
    for locus in loci.itertuples(index=False):
        block = targets[targets["target_gene"].eq(str(locus.gene).upper()) & targets["eligible_regulator"]]
        gate_block = gates[gates["locus"].eq(locus.locus)]
        grade = grades[grades["locus"].eq(locus.locus)].iloc[0]
        records.append({
            "locus": locus.locus,
            "gene": locus.gene,
            "trait": locus.trait,
            "target_states": len(block),
            "response_edges": int(block["significant_response_edges"].sum()),
            "tier_1_edges": int(block["tier_1_edges"].sum()),
            "tier_2_edges": int(block["tier_2_edges"].sum()),
            "tier_3_edges": int(block["tier_3_edges"].sum()),
            "direct_support_fraction": float(block["direct_supported_edges"].sum() / block["significant_response_edges"].sum()) if block["significant_response_edges"].sum() else np.nan,
            "causal_gates_passed": int((gate_block["status"] == "passed").sum()),
            "causal_gates_total": len(gate_block),
            "causal_grade": grade["evidence_tier"],
            "mediation_claim": False,
            "mediation_status": "not_claimed_existing_colocalization_or_instrument_gates_failed" if not bool(grade.get("causal_chain", False)) else "eligible_only_if_instrument_assumptions_hold",
        })
    return pd.DataFrame(records)


def unavailable_table(config: dict) -> pd.DataFrame:
    rows = [
        ("TFTG corroboration", "unavailable", "no frozen versioned machine-readable snapshot was acquired"),
        ("TF occupancy outside CTCF in Rest", "unavailable", "no released state-compatible primary CD4 TF occupancy experiment was identified"),
        ("Stim8hr and Stim48hr occupancy", "unavailable", "the compatible CTCF experiment is resting-state only"),
        ("state-specific ATAC and H3K27ac change", "unavailable", "the frozen peak resources do not form a matched three-state series"),
        ("RPE1 signed direction validation", "unavailable", "the frozen benchmark output retains scalar target endpoints"),
        ("guide-resolved effect vectors", "unavailable", "the public guide table contains pair summaries rather than gene-level vectors"),
        ("guide-resolved module activity", "unavailable", "gene-level guide vectors were not released"),
        ("guide-resolved transfer gain", "unavailable", "gene-level guide vectors were not released"),
        ("guide-resolved rerouting", "unavailable", "gene-level guide vectors were not released"),
        ("guide-resolved disease convergence", "unavailable", "gene-level guide vectors were not released"),
        ("donor-by-guide validation", "unavailable", "joint donor-by-guide effects were not released"),
        ("guide-to-trans cis mediation", "underpowered", "two guide doses per target-state do not identify mediation"),
    ]
    return pd.DataFrame(rows, columns=["estimand", "status", "reason"])


def summarize_timing(targets: pd.DataFrame, config: dict) -> pd.DataFrame:
    metrics = {
        "direct_support_fraction": "Regulatory support proxy",
        "occupied_fraction": "Direct occupancy support",
        "supported_path_depth": "Supported cascade depth",
        "signed_direction_agreement": "Signed agreement",
        "dominant_module_fraction": "Module coherence",
        "mean_rerouting_js": "Module rerouting",
    }
    records = []
    replicates = int(config["inference"]["bootstrap_replicates"])
    eligible = targets[targets["eligible_regulator"] & targets["significant_response_edges"].gt(0)]
    for state in STATE_ORDER:
        block = eligible[eligible["culture_condition"].eq(state)]
        for offset, (column, label) in enumerate(metrics.items()):
            if column == "occupied_fraction":
                records.append({"culture_condition": state, "metric": label, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "target_states": 0, "targets": 0, "status": "unavailable_no_eligible_occupancy_matched_factor"})
                continue
            values = block[["target_gene", column]].dropna()
            if values.empty:
                records.append({"culture_condition": state, "metric": label, "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "target_states": 0, "targets": 0, "status": "unavailable"})
                continue
            rng = np.random.default_rng(int(config["seed"]) + 700 + offset + 10 * STATE_ORDER.index(state))
            grouped = values.groupby("target_gene")[column].mean().to_numpy(dtype=float)
            boot = np.array([grouped[rng.integers(0, len(grouped), len(grouped))].mean() for _ in range(replicates)])
            records.append({
                "culture_condition": state,
                "metric": label,
                "estimate": float(values[column].mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "target_states": len(values),
                "targets": values["target_gene"].nunique(),
                "status": "estimable",
            })
    return pd.DataFrame(records)


def paired_prediction_comparison(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    records = []
    for offset, (dataset, block) in enumerate(predictions.groupby("dataset", sort=True)):
        wide = block.pivot(index=["index", "target_gene", "culture_condition", "observed"], columns="model", values="predicted").reset_index()
        statistic = lambda data: r2_score(data.observed, data.regulatory_support) - r2_score(data.observed, data.covariates)
        estimate = float(statistic(wide))
        low, high, draws = bootstrap_statistic(wide, "target_gene", statistic, int(config["inference"]["bootstrap_replicates"]), int(config["seed"]) + 1700 + offset)
        p_value = float((1 + (np.abs(draws - estimate) >= abs(estimate)).sum()) / (len(draws) + 1))
        records.append({"dataset": dataset, "targets": wide.target_gene.nunique(), "target_states": len(wide), "delta_r2": estimate, "ci_low": low, "ci_high": high, "p_value": p_value, "interval": "paired_target_bootstrap_95", "selection": "fixed_ridge_no_tuning"})
    table = pd.DataFrame(records)
    table["q_value"] = bh(table.p_value.to_numpy())
    table["decision"] = np.select([table.ci_low.gt(0) & table.q_value.lt(0.05), table.ci_high.lt(0) & table.q_value.lt(0.05)], ["improved", "worsened"], default="unresolved")
    return table


def secondary_cascade_tests(rows, genes, matrix, targets, curated, motifs, matched, config):
    names = genes.gene_name.astype(str).str.upper().to_numpy()
    measured = set(names)
    columns = {gene: i for i, gene in enumerate(names)}
    eligible = targets[targets.eligible_regulator].sort_values("response_row")
    factors = sorted(set(eligible.target_gene))
    primary = curated[curated.primary_curated & curated.curated_direction.ne(0)]
    signed = {factor: dict(zip(group.response_gene, group.curated_direction)) for factor, group in primary.groupby("source_gene")}
    memberships = {}
    for factor in factors:
        if len(signed.get(factor, {})) >= int(config["eligibility"]["activity_minimum_regulon_targets"]):
            memberships[(factor, "signed_curated")] = set(signed[factor])
        if len(motifs.get(factor, set())) >= int(config["eligibility"]["motif_minimum_targets"]):
            memberships[(factor, "motif")] = motifs[factor]
    degrees = pd.DataFrame({"factor": factors, "curated_degree": [len(signed.get(factor, {})) for factor in factors], "motif_degree": [len(motifs.get(factor, set())) for factor in factors]})
    values = np.log1p(degrees[["curated_degree", "motif_degree"]].to_numpy())
    scaled = (values - values.mean(axis=0)) / np.where(values.std(axis=0) > 0, values.std(axis=0), 1)
    reference = scaled[degrees.factor.eq("EGR1")][0]
    degrees["distance_to_egr1"] = np.sqrt(np.square(scaled - reference).sum(axis=1))
    negatives = degrees[~degrees.factor.isin(["EGR1", "GATA3"])].sort_values(["distance_to_egr1", "factor"], kind="stable").head(5).copy()
    negatives["selection"] = "five_nearest_molecular_degree_controls_before_screen"
    screen_records = []
    response_lookup = {}
    for row in eligible.itertuples(index=False):
        vector = matrix.getrow(int(row.response_row))
        effects = {names[column]: float(effect) for column, effect in zip(vector.indices, vector.data)}
        response_lookup[(row.target_gene, row.culture_condition)] = effects
        response_set = set(effects) - {row.target_gene}
        background = measured - {row.target_gene}
        for (factor, resource), full_set in sorted(memberships.items()):
            if factor == row.target_gene:
                continue
            gene_set = full_set & background
            overlap = response_set & gene_set
            a, b = len(overlap), len(response_set - gene_set)
            c, d = len(gene_set - response_set), len(background - response_set - gene_set)
            odds, p_value = fisher_exact([[a, b], [c, d]])
            intermediate = effects.get(factor, np.nan)
            signed_genes = overlap & set(signed.get(factor, {})) if resource == "signed_curated" else set()
            agreements = [np.sign(effects[gene]) == np.sign(intermediate) * signed[factor][gene] for gene in signed_genes] if np.isfinite(intermediate) else []
            screen_records.append({
                "response_row": int(row.response_row), "target_gene": row.target_gene, "culture_condition": row.culture_condition,
                "secondary_factor": factor, "resource_type": resource, "response_genes": len(response_set), "measured_distal_genes": len(background),
                "regulon_genes": len(gene_set), "overlap_genes": a, "odds_ratio": float(odds), "p_value": float(p_value),
                "secondary_retained_effect": intermediate, "secondary_response_status": "retained_significant_response" if np.isfinite(intermediate) else "no_retained_significant_response",
                "signed_edges": len(agreements), "signed_agreement": float(np.mean(agreements)) if agreements else np.nan,
                "occupancy_status": "unavailable", "screen_status": "exploratory_competitive_enrichment" if len(response_set) >= 20 else "underpowered_response_programme",
            })
    screen = pd.DataFrame(screen_records)
    screen["q_value"] = bh(screen.p_value.to_numpy())
    screen["enriched"] = screen.q_value.lt(float(config["inference"]["fdr"])) & screen.odds_ratio.gt(1)
    focus_sources = {"GATA3"} | set(negatives.factor)
    focus_factors = {"EGR1"} | set(negatives.factor)
    null_records, focus_records, replication = [], [], []
    replicates = int(config["inference"]["matched_permutation_replicates"])
    for response_row, block in matched[matched.target_gene.isin(focus_sources)].groupby("response_row", sort=True):
        source, state = block[["target_gene", "culture_condition"]].iloc[0]
        pivot = block.pivot(index="match_id", columns="slot", values="matched_gene").reindex(columns=range(6)).dropna()
        gene_array = pivot.to_numpy()
        rng = np.random.default_rng(int(config["seed"]) + 2000 + int(response_row))
        choices = rng.integers(0, 6, size=(replicates, len(gene_array)))
        for factor in sorted(focus_factors - {source}):
            for resource in ["motif", "signed_curated"]:
                gene_set = memberships.get((factor, resource))
                if gene_set is None:
                    focus_records.append({"target_gene": source, "culture_condition": state, "secondary_factor": factor, "resource_type": resource, "matched_cases": len(gene_array), "status": "unavailable_regulon_below_frozen_minimum"})
                    continue
                values = np.isin(gene_array, sorted(gene_set)).astype(float)
                delta = values[:, 0] - values[:, 1:].mean(axis=1)
                observed = float(delta.mean())
                total = values.sum(axis=1)
                selected = values[np.arange(len(values))[None, :], choices]
                null = (selected - (total[None, :] - selected) / 5).mean(axis=1)
                p_value = float((1 + (np.abs(null) >= abs(observed) - 1e-12).sum()) / (replicates + 1))
                focus_records.append({"target_gene": source, "culture_condition": state, "secondary_factor": factor, "resource_type": resource, "matched_cases": len(values), "observed_delta": observed, "null_mean": float(null.mean()), "null_low": float(np.quantile(null, 0.025)), "null_high": float(np.quantile(null, 0.975)), "p_value": p_value, "status": "estimable" if len(values) >= 20 else "underpowered"})
                null_records.extend({"target_gene": source, "culture_condition": state, "secondary_factor": factor, "resource_type": resource, "replicate": i + 1, "null_delta": float(value)} for i, value in enumerate(null))
        for factor in sorted(focus_factors - {source}):
            source_effects = response_lookup.get((source, state), {})
            factor_effects = response_lookup.get((factor, state))
            n_secondary = len(factor_effects) if factor_effects is not None else 0
            status = "eligible" if min(len(source_effects), n_secondary) >= 20 else "underpowered" if factor_effects is not None else "unavailable_perturbation_state"
            genes_common = sorted(set(source_effects) & set(factor_effects or {}))
            sign_agreement = float(np.mean([np.sign(source_effects[g]) == np.sign(factor_effects[g]) for g in genes_common])) if genes_common else np.nan
            replication.append({"target_gene": source, "culture_condition": state, "secondary_factor": factor, "source_programme_genes": len(source_effects), "secondary_programme_genes": n_secondary, "shared_retained_genes": len(genes_common), "descriptive_sign_agreement": sign_agreement, "replication_status": status, "compatible_occupancy": False, "strong_cascade_claim": False, "limitation": "full_gene_effects_and_state_matched_secondary_occupancy_unavailable"})
    targeted = pd.DataFrame(focus_records)
    targeted["q_value"] = bh(targeted.p_value.to_numpy())
    targeted["supported_enrichment"] = targeted.status.eq("estimable") & targeted.q_value.lt(0.05) & targeted.observed_delta.gt(0)
    return screen, targeted, pd.DataFrame(null_records), negatives, pd.DataFrame(replication)


def finalize_verification() -> None:
    config = yaml.safe_load((ROOT / "config/molecular_cascade.yaml").read_text())
    targets = pd.read_parquet(OUTPUT / "target_state_evidence.parquet")
    edges = pd.read_parquet(OUTPUT / "edge_evidence_matrix.parquet")
    matched = pd.read_parquet(OUTPUT / "motif_matched_backgrounds.parquet")
    pairs = pd.read_parquet(SOURCE / "analyses/vectors/results/context_rerouting_pairs.parquet")
    state_rows = []
    for state in STATE_ORDER:
        part = pairs[pairs.left_state.eq(state) | pairs.right_state.eq(state)]
        summary = part.groupby("target_contrast").module_js_divergence.agg(["mean", "count"])
        summary["state_rerouting_js"] = summary["mean"].where(summary["count"].eq(2))
        summary["culture_condition"] = state
        state_rows.append(summary.reset_index()[["target_contrast", "culture_condition", "state_rerouting_js"]])
    targets = targets.drop(columns=["state_rerouting_js"], errors="ignore").merge(pd.concat(state_rows), on=["target_contrast", "culture_condition"], how="left", validate="one_to_one")
    targets.to_parquet(OUTPUT / "target_state_evidence.parquet", index=False)
    eligible = targets[targets.eligible_regulator & targets.significant_response_edges.gt(0)]
    summaries, changes, nulls = [], [], []
    metrics = {"direct_support_fraction": "Regulatory support proxy", "supported_path_depth": "Supported cascade depth", "signed_direction_agreement": "Signed agreement", "dominant_module_fraction": "Module coherence", "state_rerouting_js": "Module rerouting", "transfer_residual": "Transfer gain"}
    replicates = int(config["inference"]["bootstrap_replicates"])
    for offset, (column, label) in enumerate(metrics.items()):
        wide = eligible.pivot_table(index="target_gene", columns="culture_condition", values=column).reindex(columns=STATE_ORDER).dropna()
        rng = np.random.default_rng(int(config["seed"]) + 4000 + offset)
        if len(wide):
            sample = rng.integers(0, len(wide), size=(replicates, len(wide)))
            for state in STATE_ORDER:
                values = wide[state].to_numpy()
                boot = values[sample].mean(axis=1)
                summaries.append({"culture_condition": state, "metric": label, "estimate": float(values.mean()), "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)), "targets": len(wide), "target_states": len(wide), "status": "estimable" if len(wide) >= 20 else "underpowered", "cohort": "identical_complete_state_targets_with_finite_metric"})
            for first, second in [("Rest", "Stim8hr"), ("Stim8hr", "Stim48hr"), ("Rest", "Stim48hr")]:
                difference = (wide[second] - wide[first]).to_numpy()
                observed = float(difference.mean())
                boot = difference[sample].mean(axis=1)
                null = (rng.choice([-1, 1], size=(replicates, len(wide))) * difference[None, :]).mean(axis=1)
                p = float((1 + (np.abs(null) >= abs(observed) - 1e-12).sum()) / (replicates + 1))
                changes.append({"metric": label, "first_state": first, "second_state": second, "targets": len(wide), "mean_difference": observed, "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)), "p_value": p, "design": "matched_target_state_label_swap_not_longitudinal_causation"})
                nulls.extend({"metric": label, "first_state": first, "second_state": second, "replicate": i + 1, "null_difference": float(value)} for i, value in enumerate(null))
    for state in STATE_ORDER:
        summaries.append({"culture_condition": state, "metric": "Direct occupancy support", "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "targets": 0, "target_states": 0, "status": "unavailable_no_eligible_occupancy_matched_factor", "cohort": "none"})
    summary = pd.DataFrame(summaries)
    change = pd.DataFrame(changes)
    change["q_value"] = bh(change.p_value.to_numpy())
    change["decision"] = np.where(change.targets.lt(20), "underpowered", np.where(change.q_value.lt(0.05) & (change.ci_low.gt(0) | change.ci_high.lt(0)), "supported", "unresolved"))
    summary.to_csv(OUTPUT / "timing_summary.csv", index=False)
    change.to_csv(OUTPUT / "paired_state_changes.csv", index=False)
    pd.DataFrame(nulls).to_parquet(OUTPUT / "state_label_permutation_null.parquet", index=False)
    coverage = edges.groupby(["response_row", "target_gene", "culture_condition"]).size().rename("response_edges").reset_index()
    counts = matched[matched.role.eq("case")].groupby("response_row").size()
    coverage["matched_response_genes"] = coverage.response_row.map(counts).fillna(0).astype(int)
    coverage["unmatched_response_genes"] = coverage.response_edges - coverage.matched_response_genes
    coverage["matched_fraction"] = coverage.matched_response_genes / coverage.response_edges
    coverage.to_csv(OUTPUT / "matched_background_coverage.csv", index=False)
    prediction = pd.read_parquet(OUTPUT / "reproducibility_oof_predictions.parquet")
    influence = []
    for dataset, block in prediction.groupby("dataset"):
        wide = block.pivot(index=["index", "target_gene", "observed"], columns="model", values="predicted").reset_index()
        baseline = r2_score(wide.observed, wide.regulatory_support) - r2_score(wide.observed, wide.covariates)
        for target in sorted(wide.target_gene.unique()):
            remaining = wide[wide.target_gene.ne(target)]
            estimate = r2_score(remaining.observed, remaining.regulatory_support) - r2_score(remaining.observed, remaining.covariates)
            influence.append({"dataset": dataset, "excluded_target": target, "targets_remaining": remaining.target_gene.nunique(), "delta_r2": float(estimate), "change_from_complete": float(estimate - baseline), "refitted": False})
    pd.DataFrame(influence).to_csv(OUTPUT / "prediction_target_influence.csv", index=False)
    resources = []
    for removed in ["none", "dorothea_a", "trrust", "regnetwork"]:
        retained_columns = [name for name in ["dorothea_a", "trrust", "regnetwork"] if name != removed]
        curated_supported = edges[retained_columns].any(axis=1)
        supported = curated_supported | edges.motif_supported | edges.promoter_bound | edges.enhancer_linked
        fractions = edges.assign(supported=supported).groupby("response_row").supported.mean()
        part = eligible.assign(regulatory_support=eligible.response_row.map(fractions))
        estimate = difference_statistic(part, "transfer_class", "amplified", "buffered", "regulatory_support")
        resources.append({"removed_resource": removed, "response_edges": len(edges), "curated_supported_edges": int(curated_supported.sum()), "regulatory_supported_edges": int(supported.sum()), "amplified_minus_buffered_support": estimate, "overlap_rule": "remaining_primary_resource_union_not_independent_votes"})
    pd.DataFrame(resources).to_csv(OUTPUT / "leave_one_curated_resource.csv", index=False)
    verification = {"frozen_inputs_verified": len(verify_freeze()), "matched_sets_with_five_controls": bool(matched.groupby("match_id").size().eq(6).all()), "caliper_enforced": bool(matched.match_distance.le(3).all()), "molecular_tiers_exclude_validation_outcomes": True, "state_summary_uses_identical_targets": True, "state_rerouting_requires_two_evaluable_pairs": True, "edge_effect_unit": "log2_fold_change_divided_by_max_absolute_cis_effect_or_0.1", "source_values_changed": False}
    (OUTPUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    full_activity_export(config, targets)


def full_activity_export(config: dict, targets: pd.DataFrame) -> None:
    genes = pd.read_parquet(SOURCE / "data/interim/gwt_vectors/genes.parquet")
    matrix = sparse.load_npz(SOURCE / "data/interim/gwt_vectors/normalized_significant_logfc.npz").tocsr()
    curated = pd.read_parquet(OUTPUT / "curated_regulatory_edges.parquet")
    network = curated[curated.dorothea_a & curated.curated_direction.ne(0)]
    minimum = int(config["eligibility"]["activity_minimum_regulon_targets"])
    counts = network.groupby("source_gene").response_gene.nunique()
    factors = sorted(counts[counts.ge(minimum)].index)
    gene_index = {gene.upper(): i for i, gene in enumerate(genes.gene_name)}
    eligible = targets[targets.eligible_regulator].sort_values("response_row")
    responses = matrix[eligible.response_row.to_numpy()]
    n = matrix.shape[1]
    response_sum = np.asarray(responses.sum(axis=1)).ravel()
    response_ss = np.asarray(responses.power(2).sum(axis=1)).ravel() - response_sum ** 2 / n
    results = []
    for factor in factors:
        block = network[network.source_gene.eq(factor)]
        weights = np.zeros(n)
        for edge in block.itertuples(index=False):
            weights[gene_index[edge.response_gene]] = edge.curated_direction
        product = np.asarray(responses @ weights).ravel()
        weight_ss = np.square(weights).sum() - weights.sum() ** 2 / n
        covariance = product - response_sum * weights.sum() / n
        correlation = covariance / np.sqrt(np.maximum(response_ss * weight_ss, 1e-24))
        correlation = np.clip(correlation, -1 + 1e-12, 1 - 1e-12)
        statistic = correlation * np.sqrt((n - 2) / np.maximum(1 - correlation ** 2, 1e-24))
        summary = eligible[["response_row", "target_gene", "culture_condition"]].copy()
        summary["inferred_tf"] = factor
        summary["regulon_genes"] = np.count_nonzero(weights)
        summary["weighted_mean_score"] = product / max(np.abs(weights).sum(), 1)
        summary["ulm_score"] = statistic
        summary["evidence_status"] = "descriptive_thresholded_response_activity_proxy"
        results.append(summary)
    table = pd.concat(results, ignore_index=True)
    table.to_parquet(OUTPUT / "all_tf_activity.parquet", index=False)
    negative = pd.read_csv(OUTPUT / "matched_negative_factors.csv").factor.tolist()
    focus = table[table.target_gene.isin(["GATA3"] + negative) & table.inferred_tf.isin(["EGR1"] + negative)].copy()
    focus["activity_inference_status"] = "no_gene_independence_or_full_effect_assumption_for_significance"
    focus.to_csv(OUTPUT / "targeted_tf_activity.csv", index=False)


def reconcile_permutation_ties() -> None:
    specifications = [
        ("paired_state_changes.csv", "state_label_permutation_null.parquet", ["metric", "first_state", "second_state"], "mean_difference", "null_difference"),
        ("targeted_cascade_enrichment.csv", "targeted_cascade_permutation_null.parquet", ["target_gene", "culture_condition", "secondary_factor", "resource_type"], "observed_delta", "null_delta"),
    ]
    for filename, null_file, keys, estimate, null_column in specifications:
        table = pd.read_csv(OUTPUT / filename)
        groups = pd.read_parquet(OUTPUT / null_file).groupby(keys)[null_column]
        for i, row in table.iterrows():
            if not np.isfinite(row.get(estimate, np.nan)):
                continue
            values = groups.get_group(tuple(row[key] for key in keys)).to_numpy()
            table.loc[i, "p_value"] = (1 + (np.abs(values) >= abs(row[estimate]) - 1e-12).sum()) / (len(values) + 1)
        table["q_value"] = bh(table.p_value.to_numpy())
        if filename.startswith("paired_state"):
            table["decision"] = np.where(table.targets.lt(20), "underpowered", np.where(table.q_value.lt(0.05) & (table.ci_low.gt(0) | table.ci_high.lt(0)), "supported", "unresolved"))
        else:
            table["supported_enrichment"] = table.status.eq("estimable") & table.q_value.lt(0.05) & table.observed_delta.gt(0)
        table.to_csv(OUTPUT / filename, index=False)


def main() -> None:
    config = yaml.safe_load((ROOT / "config/molecular_cascade.yaml").read_text(encoding="utf-8"))
    frozen_checks = verify_freeze()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(SOURCE / "data/interim/gwt_vectors/rows.parquet").reset_index(drop=True)
    rows["target_contrast_gene_name"] = rows["target_contrast_gene_name"].astype(str).str.upper()
    genes = pd.read_parquet(SOURCE / "data/interim/gwt_vectors/genes.parquet").reset_index(drop=True)
    genes["gene_name"] = genes["gene_name"].astype(str).str.upper()
    response_matrix = sparse.load_npz(SOURCE / "data/interim/gwt_vectors/normalized_significant_logfc.npz").tocsr()
    if response_matrix.shape != (len(rows), len(genes)):
        raise ValueError("Frozen response matrix dimensions disagree with metadata")
    measured = set(genes["gene_name"])
    beds = {
        "atac": read_bed(SOURCE / "work/causal_inputs/ENCFF944LFH.bed.gz"),
        "h3k27ac": read_bed(SOURCE / "work/causal_inputs/ENCFF068XUG.bed.gz"),
        "ctcf": read_bed(SOURCE / "work/regulatory_inputs/ENCFF858TLX.bed.gz"),
    }
    gene_table = build_gene_table(genes, config, beds)
    curated = load_curated(SOURCE / "work/regulatory_inputs/omnipath_tf_target.tsv", measured)
    motifs = load_motifs(
        SOURCE / "work/motif/TRANSFAC_and_JASPAR_PWMs.gmt",
        measured,
        int(config["eligibility"]["motif_minimum_targets"]),
        int(config["eligibility"]["motif_maximum_targets"]),
    )
    pchic_cache = OUTPUT / "pchic_lifted_links.parquet"
    pchic_audit_cache = OUTPUT / "pchic_liftover_audit.json"
    if pchic_cache.exists() and pchic_audit_cache.exists():
        pchic_links = pd.read_parquet(pchic_cache)
        pchic_audit = json.loads(pchic_audit_cache.read_text(encoding="utf-8"))
    else:
        pchic_links, pchic_audit = prepare_pchic(config, measured, beds)
        pchic_links.to_parquet(pchic_cache, index=False)
        pchic_audit_cache.write_text(json.dumps(pchic_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pchic_primary = pchic_support(pchic_links, config)
    guides = guide_summary(config)
    k562 = pd.read_parquet(SOURCE / "analyses/replication/results/k562_replication_rows.parquet")
    k562 = k562.rename(columns={"target_contrast_gene_name": "target_gene"})
    k562["culture_condition"] = k562["condition"]
    k562["target_gene"] = k562["target_gene"].astype(str).str.upper()
    k562 = k562[["target_gene", "culture_condition", "logfc_pearson_r", "logfc_pearson_pval", "correlation_delta", "shared_response", "private_or_inverted", "confidence_tier"]].drop_duplicates(["target_gene", "culture_condition"])

    edges, targets = build_edge_matrix(rows, genes, response_matrix, curated, motifs, pchic_primary, gene_table, guides, k562)
    curated_columns = ["source_gene", "response_gene", "resources", "dorothea_a", "dorothea_b", "trrust", "regnetwork", "collectri"]
    edges = edges.merge(curated[curated_columns], left_on=["target_gene", "response_gene"], right_on=["source_gene", "response_gene"], how="left")
    edges = edges.drop(columns=["source_gene"])
    for column in ["dorothea_a", "dorothea_b", "trrust", "regnetwork", "collectri"]:
        edges[column] = edges[column].eq(True)
    edges["resources"] = edges["resources"].fillna("")

    phenotype = pd.read_parquet(SOURCE / "analyses/primary/results/transfer_phenotypes.parquet")
    phenotype_columns = ["index", "cis_magnitude", "log1p_target_baseMean", "log1p_n_cells_target", "log1p_n_downstream", "transfer_residual", "transfer_z", "transfer_class", "cluster"]
    targets = targets.merge(phenotype[phenotype_columns], on="index", how="left", validate="one_to_one")
    targets = targets.merge(guides.drop(columns=["guide_concordant"]), on=["target_gene", "culture_condition"], how="left", validate="one_to_one")
    targets = targets.merge(k562, on=["target_gene", "culture_condition"], how="left", validate="one_to_one")
    gain = pd.read_parquet(SOURCE / "analyses/vectors/results/network_gain_rows.parquet")
    targets = targets.merge(gain[["index", "dominant_module", "dominant_module_fraction", "oof_reconstruction_cosine"]], on="index", how="left", validate="one_to_one")
    rerouting = pd.read_parquet(SOURCE / "analyses/vectors/results/context_rerouting_pairs.parquet").groupby("target_contrast", as_index=False)["module_js_divergence"].mean().rename(columns={"module_js_divergence": "mean_rerouting_js"})
    targets = targets.merge(rerouting, on="target_contrast", how="left", validate="many_to_one")

    disease = pd.read_parquet(SOURCE / "analyses/disease/results/disease_model_rows.parquet")
    disease = disease[disease["significant"] & ~disease["negative_control_disease"]].copy()
    disease["culture_condition"] = disease["gene_set"].str.extract(r"(Rest|Stim8hr|Stim48hr)")
    disease_keys = set(zip(disease["cluster"], disease["culture_condition"], strict=True))
    targets["disease_program_convergence"] = [(cluster, state) in disease_keys if pd.notna(cluster) else False for cluster, state in targets[["cluster", "culture_condition"]].itertuples(index=False, name=None)]

    eligible_rows = targets.loc[targets["eligible_regulator"], "response_row"].to_numpy(dtype=int)
    activity_top, source_activity, activity_sensitivity = infer_tf_activity(rows, response_matrix, genes, curated, eligible_rows, config)
    targets = targets.merge(source_activity, on="response_row", how="left", validate="one_to_one")
    activity_range = targets.groupby("target_gene")["source_tf_activity"].agg(lambda values: values.max() - values.min() if values.notna().sum() >= 2 else np.nan)
    targets["source_tf_activity_range"] = targets["target_gene"].map(activity_range)

    matched_cache = OUTPUT / "motif_matched_backgrounds.parquet"
    matched_cache_meta = OUTPUT / "matched_background_cache.json"
    matched_key = hashlib.sha256((sha256(ROOT / "config/molecular_cascade.yaml") + sha256(OUTPUT / "extension_rules.json") + str(len(edges)) + str(int(edges["response_row"].sum()))).encode()).hexdigest()
    if matched_cache.exists() and matched_cache_meta.exists() and json.loads(matched_cache_meta.read_text(encoding="utf-8")).get("key") == matched_key:
        matched = pd.read_parquet(matched_cache)
    else:
        matched = build_matched_backgrounds(edges, rows, gene_table, curated, motifs, config)
        matched.to_parquet(matched_cache, index=False)
        matched_cache_meta.write_text(json.dumps({"key": matched_key, "rows": len(matched)}, indent=2) + "\n", encoding="utf-8")
    matched_permutations, matched_summary = run_matched_permutations(matched, config)
    rewiring = pd.read_parquet(OUTPUT / "degree_preserving_rewiring_null.parquet") if (OUTPUT / "degree_preserving_rewiring_null.parquet").exists() else degree_preserving_null(edges, curated, config)
    ablation = component_ablation(edges, targets)
    threshold = threshold_sensitivity(pchic_links, gene_table, edges, config)
    influence = leave_one_influence(targets, ablation)

    natural = pd.read_parquet(SOURCE / "analyses/natural_genetics/results/directional_pairs.parquet")
    natural = natural.merge(edges[["response_row", "gene_column", "target_gene", "response_gene", "edge_class", "evidence_tier", "direction_status"]], on=["response_row", "gene_column"], how="inner", validate="many_to_one")
    predictions, predictive_metrics = run_predictive_validation(targets, config)
    hypotheses = compile_hypotheses(targets, edges, rewiring, natural, config)
    central = central_verification(hypotheses, targets)
    loci = locus_verification(targets)
    timing = summarize_timing(targets, config)
    guide_tiers = guide_tier_summary(targets, config)
    unavailable = unavailable_table(config)
    secondary, targeted, targeted_null, negative_factors, cascade_gates = secondary_cascade_tests(rows, genes, response_matrix, targets, curated, motifs, matched, config)
    prediction_comparison = paired_prediction_comparison(predictions, config)

    motif_rows = []
    eligible_sources = set(targets.loc[targets["eligible_regulator"], "target_gene"])
    for source in sorted(eligible_sources & set(motifs)):
        for response_gene in sorted(motifs[source]):
            motif_rows.append({"source_gene": source, "response_gene": response_gene, "resource": "TRANSFAC_and_JASPAR_PWMs_human_terms"})
    motif_membership = pd.DataFrame(motif_rows)

    primary_edge_total = int(edges["curated_edge"].sum())
    tier_counts = edges["evidence_tier"].value_counts().to_dict()
    class_counts = edges["edge_class"].value_counts().to_dict()
    verification_rows = targets[targets["eligible_regulator"]].sort_values("index", kind="stable").head(25)
    verification = []
    for row in verification_rows.itertuples(index=False):
        matrix_count = int(response_matrix.getrow(int(row.response_row)).nnz)
        evidence_count = int((edges["response_row"] == row.response_row).sum())
        if matrix_count != evidence_count or matrix_count != row.significant_response_edges:
            raise ValueError(f"Edge denominator mismatch for {row.index}")
        verification.append({"index": row.index, "response_row": int(row.response_row), "matrix_nonzero": matrix_count, "evidence_rows": evidence_count, "match": True})
    if (edges["evidence_tier"].eq("Tier 1") & ~(edges["enhancer_linked"] & edges["physical_active_link"] & edges["guide_concordant"] & ~edges["directional_contradiction"])).any():
        raise ValueError("Tier 1 gate was not enforced")

    gene_table.to_parquet(OUTPUT / "gene_regulatory_covariates.parquet", index=False)
    curated.to_parquet(OUTPUT / "curated_regulatory_edges.parquet", index=False)
    motif_membership.to_parquet(OUTPUT / "motif_membership.parquet", index=False)
    pchic_links.to_parquet(OUTPUT / "pchic_lifted_links.parquet", index=False)
    edges.to_parquet(OUTPUT / "edge_evidence_matrix.parquet", index=False)
    targets.to_parquet(OUTPUT / "target_state_evidence.parquet", index=False)
    activity_top.to_parquet(OUTPUT / "tf_activity_top.parquet", index=False)
    activity_sensitivity.to_csv(OUTPUT / "tf_activity_sensitivity.csv", index=False)
    matched.to_parquet(OUTPUT / "motif_matched_backgrounds.parquet", index=False)
    matched_permutations.to_parquet(OUTPUT / "matched_permutation_null.parquet", index=False)
    matched_summary.to_csv(OUTPUT / "matched_permutation_summary.csv", index=False)
    rewiring.to_parquet(OUTPUT / "degree_preserving_rewiring_null.parquet", index=False)
    ablation.to_csv(OUTPUT / "component_ablation.csv", index=False)
    threshold.to_csv(OUTPUT / "threshold_sensitivity.csv", index=False)
    influence.to_parquet(OUTPUT / "leave_one_influence.parquet", index=False)
    natural.to_parquet(OUTPUT / "natural_genetic_edge_verification.parquet", index=False)
    predictions.to_parquet(OUTPUT / "reproducibility_oof_predictions.parquet", index=False)
    predictive_metrics.to_csv(OUTPUT / "predictive_validation.csv", index=False)
    hypotheses.to_csv(OUTPUT / "hypothesis_decisions.csv", index=False)
    central.to_csv(OUTPUT / "central_result_verification.csv", index=False)
    loci.to_csv(OUTPUT / "locus_regulatory_verification.csv", index=False)
    timing.to_csv(OUTPUT / "timing_summary.csv", index=False)
    guide_tiers.to_csv(OUTPUT / "guide_concordance_by_tier.csv", index=False)
    unavailable.to_csv(OUTPUT / "unavailable_estimands.csv", index=False)
    secondary.to_parquet(OUTPUT / "secondary_factor_screen.parquet", index=False)
    targeted.to_csv(OUTPUT / "targeted_cascade_enrichment.csv", index=False)
    targeted_null.to_parquet(OUTPUT / "targeted_cascade_permutation_null.parquet", index=False)
    negative_factors.to_csv(OUTPUT / "matched_negative_factors.csv", index=False)
    cascade_gates.to_csv(OUTPUT / "cascade_replication_gates.csv", index=False)
    prediction_comparison.to_csv(OUTPUT / "paired_prediction_comparison.csv", index=False)
    (OUTPUT / "frozen_input_verification.json").write_text(json.dumps(frozen_checks, indent=2) + "\n")
    (OUTPUT / "deterministic_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    audit = {
        "design_status": config["design"]["status"],
        "primary_target_state_denominator": len(targets),
        "eligible_regulator_target_states": int(targets["eligible_regulator"].sum()),
        "eligible_regulators": int(targets.loc[targets["eligible_regulator"], "target_gene"].nunique()),
        "eligible_null_response_target_states": int((targets["eligibility_status"] == "eligible_null_response").sum()),
        "response_edge_denominator": len(edges),
        "primary_curated_edges_observed": primary_edge_total,
        "edge_class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "evidence_tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
        "direct_regulation_claims": int(tier_counts.get("Tier 1", 0)),
        "pchic": pchic_audit,
        "motif_regulators": len(motifs),
        "curated_regulators": int(curated.loc[curated["primary_curated"], "source_gene"].nunique()),
        "matched_case_control_rows": len(matched),
        "matched_permutation_replicates": int(config["inference"]["matched_permutation_replicates"]),
        "degree_rewiring_replicates": len(rewiring),
        "hypothesis_decisions": hypotheses["decision"].value_counts().to_dict(),
        "unavailable_estimands": len(unavailable),
        "deterministic_rows_reproduced": len(verification),
        "source_freeze_sha256": sha256(ROOT / "analyses/molecular_cascade/freeze_manifest.json"),
        "extension_rules_sha256": sha256(OUTPUT / "extension_rules.json"),
        "direct_support_fraction_definition": "legacy column name: motif, curated, promoter or enhancer support; not proof of direct regulation",
        "activity_limitation": "Scores use the frozen significant-effect matrix; absent entries are thresholded, not full measured effects.",
        "secondary_screen_rows": len(secondary),
        "secondary_screen_status": "exploratory_complete_denominator",
    }
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finalize_verification()


if __name__ == "__main__":
    main()
