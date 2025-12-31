"""Evaluate every frozen locus against the causal-triangulation gates."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/causal_triangulation/results"
CONFIG = ROOT / "config/causal_triangulation.yaml"
ATAC = ROOT / "work/causal_inputs/ENCFF944LFH.bed.gz"
H3K27AC = ROOT / "work/causal_inputs/ENCFF068XUG.bed.gz"
LOCUS_ORDER = ["gata3", "stat3", "ptpn22"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    if len(array) == 0:
        return np.array([], dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def load_bed(path: Path) -> dict[str, list[tuple[int, int]]]:
    peaks: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            chromosome = fields[0].removeprefix("chr")
            peaks.setdefault(chromosome, []).append((int(fields[1]), int(fields[2])))
    return peaks


def nearest_peak(position_0based: int, intervals: list[tuple[int, int]]) -> tuple[bool, int | None]:
    if not intervals:
        return False, None
    overlap = any(start <= position_0based < end for start, end in intervals)
    distances = [
        0
        if start <= position_0based < end
        else max(0, start - (position_0based + 1))
        if position_0based < start
        else max(0, position_0based - end)
        for start, end in intervals
    ]
    return overlap, int(min(distances))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    alpha = float(config["analysis"]["alpha"])
    coloc_threshold = float(
        config["evidence_gates"]["gwas_cis_eqtl_colocalization"][
            "minimum_shared_variant_posterior"
        ]
    )
    minimum_links = int(
        config["evidence_gates"]["enhancer_to_gene"][
            "otherwise_minimum_concordant_orthogonal_links"
        ]
    )

    summary = pd.read_csv(ROOT / "analyses/loci/results/locus_summary.csv")
    variants = pd.read_csv(ROOT / "analyses/loci/results/credible_set_variants.csv")
    features = pd.read_csv(ROOT / "analyses/loci/results/l2g_features.csv")
    programs = pd.read_csv(ROOT / "analyses/loci/results/transfer_programs.csv")
    pchic = pd.read_csv(ROOT / "analyses/loci/results/pchic_locus_summary.csv").set_index("locus")
    phenotypes = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    cis = pd.read_parquet(ROOT / "data/interim/eqtlgen/cis_mediators.parquet")
    trans = pd.read_parquet(ROOT / "data/interim/eqtlgen/trans_associations.parquet")
    peak_sets = {"ATAC": load_bed(ATAC), "H3K27ac": load_bed(H3K27AC)}

    feature_map = {
        (str(row.locus), str(row.feature)): float(row.value) if pd.notna(row.value) else np.nan
        for row in features.itertuples(index=False)
    }
    contexts: list[dict[str, object]] = []
    direction_records: list[dict[str, object]] = []
    instrument_records: list[dict[str, object]] = []
    deletion_records: list[dict[str, object]] = []
    chromatin_records: list[dict[str, object]] = []
    enhancer_records: list[dict[str, object]] = []
    coloc_records: list[dict[str, object]] = []

    for locus in LOCUS_ORDER:
        meta = summary.loc[summary["locus"] == locus].iloc[0]
        gene = str(meta["gene"])
        lead = str(meta["lead_rsid"])
        locus_variants = variants[
            (variants["locus"] == locus) & variants["is_95_credible_set"]
        ].copy()

        assay_overlap_ids: dict[str, set[str]] = {}
        assay_posterior: dict[str, float] = {}
        assay_distances: dict[str, list[int]] = {}
        for assay, peaks in peak_sets.items():
            overlap_ids: set[str] = set()
            posterior = 0.0
            distances: list[int] = []
            for variant in locus_variants.itertuples(index=False):
                overlap, distance = nearest_peak(
                    int(variant.position) - 1,
                    peaks.get(str(variant.chromosome).removeprefix("chr"), []),
                )
                if overlap:
                    overlap_ids.add(str(variant.variant_id))
                    posterior += float(variant.posterior_probability)
                if distance is not None:
                    distances.append(distance)
                chromatin_records.append(
                    {
                        "locus": locus,
                        "gene": gene,
                        "variant_id": variant.variant_id,
                        "posterior_probability": variant.posterior_probability,
                        "assay": assay,
                        "overlap": overlap,
                        "nearest_peak_bp": distance,
                    }
                )
            assay_overlap_ids[assay] = overlap_ids
            assay_posterior[assay] = posterior
            assay_distances[assay] = distances
        chromatin_ids = assay_overlap_ids["ATAC"] | assay_overlap_ids["H3K27ac"]
        chromatin_pass = bool(chromatin_ids)
        chromatin_posterior = float(
            locus_variants.loc[
                locus_variants["variant_id"].isin(chromatin_ids), "posterior_probability"
            ].sum()
        )

        target_features = {
            name: feature_map.get((locus, name), np.nan)
            for name in ["eQtlColocH4Maximum", "eQtlColocClppMaximum", "e2gMean"]
        }
        coloc_value = target_features["eQtlColocH4Maximum"]
        shared_posterior_pass = bool(
            np.isfinite(coloc_value) and coloc_value >= coloc_threshold
        )
        regional_coverage_status = "unavailable"
        coloc_status = "failed" if not shared_posterior_pass else "unresolved"
        coloc_records.append(
            {
                "locus": locus,
                "gene": gene,
                "shared_variant_posterior": coloc_value,
                "minimum_shared_variant_posterior": coloc_threshold,
                "shared_variant_posterior_pass": shared_posterior_pass,
                "regional_coverage_status": regional_coverage_status,
                "final_status": coloc_status,
                "reason": (
                    "shared-variant posterior below threshold"
                    if not shared_posterior_pass
                    else "posterior threshold passed, but adequate regional coverage was not reported by the frozen source"
                ),
            }
        )

        e2g_value = target_features["e2gMean"]
        pchic_contact_variants = int(pchic.loc[locus, "variants_with_called_contact"])
        pchic_nearest = pchic.loc[locus, "nearest_called_contact_bp"]
        link_states = {
            "activity_by_contact": False,
            "promoter_capture_hic": pchic_contact_variants > 0,
            "coaccessibility": False,
            "curated_locus_to_gene": False,
        }
        link_status = {
            "activity_by_contact": (
                "available_unqualified"
                if np.isfinite(e2g_value) and e2g_value > 0
                else "unsupported"
            ),
            "promoter_capture_hic": (
                "supported" if pchic_contact_variants > 0 else "unsupported"
            ),
            "coaccessibility": "unavailable",
            "curated_locus_to_gene": "unavailable",
        }
        link_values = {
            "activity_by_contact": e2g_value,
            "promoter_capture_hic": pchic_contact_variants,
            "coaccessibility": np.nan,
            "curated_locus_to_gene": np.nan,
        }
        link_reasons = {
            "activity_by_contact": "Open Targets rE2G aggregate feature",
            "promoter_capture_hic": "credible variants in called primary CD4 PCHiC interactions",
            "coaccessibility": "no frozen locus-specific coaccessibility link",
            "curated_locus_to_gene": "no independent frozen curated enhancer-to-gene link",
        }
        for link_type in link_states:
            enhancer_records.append(
                {
                    "locus": locus,
                    "gene": gene,
                    "link_type": link_type,
                    "status": link_status[link_type],
                    "value": link_values[link_type],
                    "evidence": link_reasons[link_type],
                }
            )
        enhancer_records.append(
            {
                "locus": locus,
                "gene": gene,
                "link_type": "experimental_enhancer_perturbation",
                "status": "unavailable",
                "value": np.nan,
                "evidence": "no frozen enhancer perturbation assay at this locus",
            }
        )
        supported_links = sum(link_states.values())
        enhancer_pass = supported_links >= minimum_links

        perturb = phenotypes[phenotypes["target_contrast_gene_name"] == gene].copy()
        cis_pass = bool(
            len(perturb) > 0 and perturb["ontarget_significant"].fillna(False).all()
        )
        program = programs[programs["locus"] == locus].copy()
        program_nonzero = program["normalized_signed_effect"].fillna(0).ne(0)
        program_pass = bool(len(program) > 0 and program_nonzero.any())

        lead_cis = cis[(cis["SNP"] == lead) & (cis["GeneSymbol"] == gene)].copy()
        lead_trans = trans[trans["SNP"] == lead].copy()
        if len(lead_cis):
            for row in lead_cis.itertuples(index=False):
                instrument_records.append(
                    {
                        "locus": locus,
                        "gene": gene,
                        "rsid": lead,
                        "role": "variant_to_cis",
                        "assessed_allele": row.AssessedAllele,
                        "other_allele": row.OtherAllele,
                        "z_score": row.Zscore,
                        "p_value": row.Pvalue,
                        "samples": row.NrSamples,
                        "status": "observed_but_ineligible",
                        "valid_for_mediation": False,
                        "exclusion_reason": "fewer_than_three_independent_instruments",
                    }
                )
        else:
            instrument_records.append(
                {
                    "locus": locus,
                    "gene": gene,
                    "rsid": lead,
                    "role": "variant_to_cis",
                    "assessed_allele": np.nan,
                    "other_allele": np.nan,
                    "z_score": np.nan,
                    "p_value": np.nan,
                    "samples": np.nan,
                    "status": "unavailable",
                    "valid_for_mediation": False,
                    "exclusion_reason": "no_lead_variant_cis_association_for_selected_gene",
                }
            )

        merged = lead_trans.merge(
            program[["condition", "downstream_gene", "normalized_signed_effect"]],
            left_on="GeneSymbol",
            right_on="downstream_gene",
            how="inner",
        )
        if len(lead_cis) and len(perturb) and len(merged):
            cis_z = float(lead_cis.iloc[0]["Zscore"])
            perturb_sign = np.sign(float(perturb["ontarget_effect_size"].median()))
            orientation = np.sign(cis_z) / perturb_sign if perturb_sign != 0 else np.nan
            merged["predicted_variant_effect"] = merged["normalized_signed_effect"] * orientation
            merged["direction_match"] = (
                np.sign(merged["predicted_variant_effect"]) == np.sign(merged["Zscore"])
            )
            for row in merged.itertuples(index=False):
                direction_records.append(
                    {
                        "locus": locus,
                        "gene": gene,
                        "rsid": lead,
                        "condition": row.condition,
                        "distal_gene": row.GeneSymbol,
                        "trans_z": row.Zscore,
                        "program_effect": row.normalized_signed_effect,
                        "predicted_variant_effect": row.predicted_variant_effect,
                        "direction_match": row.direction_match,
                    }
                )
            matches = int(merged["direction_match"].sum())
            trials = len(merged)
            direction_p = float(binomtest(matches, trials, 0.5, alternative="greater").pvalue)
            direction_fraction = matches / trials
            for condition, block in merged.groupby("condition"):
                full_fraction = float(block["direction_match"].mean())
                for distal in block["GeneSymbol"].unique():
                    reduced = block[block["GeneSymbol"] != distal]
                    deletion_records.append(
                        {
                            "locus": locus,
                            "deletion_unit": distal,
                            "condition": condition,
                            "full_direction_fraction": full_fraction,
                            "deleted_direction_fraction": (
                                float(reduced["direction_match"].mean()) if len(reduced) else np.nan
                            ),
                            "estimate_change": (
                                float(reduced["direction_match"].mean() - full_fraction)
                                if len(reduced)
                                else np.nan
                            ),
                        }
                    )
        else:
            matches = 0
            trials = 0
            direction_p = np.nan
            direction_fraction = np.nan

        contexts.append(
            {
                "locus": locus,
                "gene": gene,
                "trait": meta["trait"],
                "lead_rsid": lead,
                "meta": meta,
                "locus_variants": locus_variants,
                "atac_ids": assay_overlap_ids["ATAC"],
                "atac_posterior": assay_posterior["ATAC"],
                "atac_distances": assay_distances["ATAC"],
                "h3k27ac_ids": assay_overlap_ids["H3K27ac"],
                "h3k27ac_posterior": assay_posterior["H3K27ac"],
                "h3k27ac_distances": assay_distances["H3K27ac"],
                "chromatin_ids": chromatin_ids,
                "chromatin_posterior": chromatin_posterior,
                "chromatin_pass": chromatin_pass,
                "coloc_value": coloc_value,
                "coloc_status": coloc_status,
                "e2g_value": e2g_value,
                "supported_links": supported_links,
                "enhancer_pass": enhancer_pass,
                "pchic_contact_variants": pchic_contact_variants,
                "pchic_nearest": pchic_nearest,
                "perturb": perturb,
                "cis_pass": cis_pass,
                "program": program,
                "program_nonzero_count": int(program_nonzero.sum()),
                "program_pass": program_pass,
                "lead_cis_count": len(lead_cis),
                "direction_matches": matches,
                "direction_trials": trials,
                "direction_fraction": direction_fraction,
                "direction_p": direction_p,
            }
        )

    locus_tests = pd.DataFrame(
        [
            {
                "locus": context["locus"],
                "matches": context["direction_matches"],
                "count": context["direction_trials"],
                "direction_fraction": context["direction_fraction"],
                "p_value": context["direction_p"],
            }
            for context in contexts
        ]
    )
    locus_tests["q_value"] = np.nan
    finite = locus_tests["p_value"].notna()
    if finite.any():
        locus_tests.loc[finite, "q_value"] = bh(locus_tests.loc[finite, "p_value"])
    locus_tests["status"] = "unresolved"
    testable = locus_tests["count"] > 0
    locus_tests.loc[testable, "status"] = "failed"
    locus_tests.loc[
        testable
        & locus_tests["direction_fraction"].gt(0.5)
        & locus_tests["q_value"].lt(alpha),
        "status",
    ] = "passed"
    trans_status = locus_tests.set_index("locus")["status"].to_dict()
    trans_q = locus_tests.set_index("locus")["q_value"].to_dict()

    gate_records: list[dict[str, object]] = []
    locus_records: list[dict[str, object]] = []
    unavailable_records: list[dict[str, object]] = []
    for context in contexts:
        locus = str(context["locus"])
        gene = str(context["gene"])
        direction_status = trans_status[locus]
        direction_evidence = (
            "no lead-variant trans associations overlapped the frozen signed program"
            if int(context["direction_trials"]) == 0
            else f"eQTLGen direction test with BH q={float(trans_q[locus]):.4g}"
        )
        gates = [
            (
                "statistical_fine_mapping",
                "passed",
                float(context["meta"]["lead_posterior_probability"]),
                "Open Targets 95% credible set",
            ),
            (
                "gwas_cis_eqtl_colocalization",
                context["coloc_status"],
                context["coloc_value"],
                "Open Targets target-specific H4 plus required regional-coverage gate",
            ),
            (
                "credible_set_cd4_chromatin_overlap",
                "passed" if context["chromatin_pass"] else "failed",
                len(context["chromatin_ids"]),
                (
                    f"95% credible set: ATAC {len(context['atac_ids'])} variants; "
                    f"H3K27ac {len(context['h3k27ac_ids'])} variants; "
                    f"combined posterior mass {float(context['chromatin_posterior']):.6f}"
                ),
            ),
            (
                "enhancer_to_gene",
                "passed" if context["enhancer_pass"] else "failed",
                context["supported_links"],
                f"{context['supported_links']} of {minimum_links} required orthogonal links; no enhancer perturbation assay",
            ),
            (
                "perturbational_cis_effect",
                "passed" if context["cis_pass"] else "failed",
                (
                    float(context["perturb"]["ontarget_effect_size"].median())
                    if len(context["perturb"])
                    else np.nan
                ),
                "frozen CD4 on-target z-score with source significance required in every state",
            ),
            (
                "signed_downstream_program",
                "passed" if context["program_pass"] else "failed",
                context["program_nonzero_count"],
                "nonzero effects in the frozen signed CD4 program",
            ),
            (
                "trans_eqtl_direction_concordance",
                direction_status,
                context["direction_fraction"],
                direction_evidence,
            ),
        ]
        for gate, status, value, evidence in gates:
            gate_records.append(
                {
                    "locus": locus,
                    "gene": gene,
                    "trait": context["trait"],
                    "gate": gate,
                    "status": status,
                    "value": value,
                    "evidence": evidence,
                }
            )

        passed = sum(status == "passed" for _, status, _, _ in gates)
        all_gates = passed == len(gates)
        directional_contradiction = direction_status == "failed"
        if all_gates:
            tier = "tier_2_convergent_without_valid_mediation"
        elif passed >= 4 and not directional_contradiction:
            tier = "tier_3_partial"
        else:
            tier = "tier_4_unsupported_or_contradictory"
        stopped = [gate for gate, status, _, _ in gates if status != "passed"]
        locus_records.append(
            {
                "locus": locus,
                "gene": gene,
                "trait": context["trait"],
                "lead_rsid": context["lead_rsid"],
                "credible_set_variants": len(context["locus_variants"]),
                "cd4_atac_overlap_variants": len(context["atac_ids"]),
                "cd4_atac_overlap_posterior": context["atac_posterior"],
                "nearest_cd4_atac_peak_bp": (
                    min(context["atac_distances"]) if context["atac_distances"] else np.nan
                ),
                "cd4_h3k27ac_overlap_variants": len(context["h3k27ac_ids"]),
                "cd4_h3k27ac_overlap_posterior": context["h3k27ac_posterior"],
                "nearest_cd4_h3k27ac_peak_bp": (
                    min(context["h3k27ac_distances"])
                    if context["h3k27ac_distances"]
                    else np.nan
                ),
                "cd4_chromatin_overlap_variants": len(context["chromatin_ids"]),
                "cd4_chromatin_overlap_posterior": context["chromatin_posterior"],
                "cd4_pchic_contact_variants": context["pchic_contact_variants"],
                "nearest_cd4_pchic_contact_bp": context["pchic_nearest"],
                "passed_gates": passed,
                "total_gates": len(gates),
                "evidence_tier": tier,
                "mediation_status": "not_tested_assumptions_failed",
                "mediation_claim": False,
                "primary_stop_reason": ";".join(stopped),
            }
        )
        unavailable_records.extend(
            [
                {
                    "analysis": "colocalization_regional_coverage",
                    "locus": locus,
                    "status": "unavailable",
                    "reason": "the frozen Open Targets feature export did not report regional variant coverage",
                },
                {
                    "analysis": "two_step_mediation",
                    "locus": locus,
                    "status": "not_tested",
                    "reason": "colocalization or minimum instrument assumptions failed",
                },
                {
                    "analysis": "matched_negative_locus",
                    "locus": locus,
                    "status": "unresolved",
                    "reason": "MAF and local gene-density fields were unavailable in the frozen source bundle",
                },
            ]
        )

    direction_columns = [
        "locus", "gene", "rsid", "condition", "distal_gene", "trans_z",
        "program_effect", "predicted_variant_effect", "direction_match",
    ]
    deletion_columns = [
        "locus", "deletion_unit", "condition", "full_direction_fraction",
        "deleted_direction_fraction", "estimate_change",
    ]
    pd.DataFrame(gate_records).to_csv(OUTPUT / "evidence_gates.csv", index=False)
    pd.DataFrame(locus_records).to_csv(OUTPUT / "locus_grades.csv", index=False)
    pd.DataFrame(instrument_records).to_csv(OUTPUT / "instrument_audit.csv", index=False)
    pd.DataFrame(direction_records, columns=direction_columns).to_csv(
        OUTPUT / "trans_direction_rows.csv", index=False
    )
    locus_tests.to_csv(OUTPUT / "trans_direction_tests.csv", index=False)
    pd.DataFrame(deletion_records, columns=deletion_columns).to_csv(
        OUTPUT / "leave_one_distal_gene.csv", index=False
    )
    pd.DataFrame(unavailable_records).to_csv(OUTPUT / "unavailable_tests.csv", index=False)
    pd.DataFrame(chromatin_records).to_csv(OUTPUT / "chromatin_overlap_rows.csv", index=False)
    pd.DataFrame(enhancer_records).to_csv(OUTPUT / "enhancer_link_components.csv", index=False)
    pd.DataFrame(coloc_records).to_csv(OUTPUT / "colocalization_gate_components.csv", index=False)

    audit = {
        "eligible_locus_denominator": len(LOCUS_ORDER),
        "loci_with_mediation_claim": 0,
        "config_sha256": sha256(CONFIG),
        "atac_source": "ENCODE ENCSR841LHT, ENCFF944LFH, GRCh38 IDR thresholded peaks",
        "atac_source_noncompliance_note": True,
        "atac_file_sha256": sha256(ATAC),
        "h3k27ac_source": "ENCODE ENCSR396RXX, ENCFF068XUG, GRCh38 pseudoreplicated narrow peaks",
        "h3k27ac_file_sha256": sha256(H3K27AC),
        "chromatin_gate_rule": "at least one 95% credible-set variant overlaps CD4 ATAC or H3K27ac",
        "colocalization_coverage_status": "unavailable_for_all_loci",
        "pchic_source": "Burren et al. 2017 Additional file 5 Table S4, lifted from GRCh37 with the frozen UCSC chain",
        "pchic_call_rule": "CHiCAGO score greater than 5 in either CD4 state",
        "all_loci_reported": True,
        "matched_negative_locus_status": "unresolved_missing_matching_fields",
    }
    (OUTPUT / "causal_triangulation_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
