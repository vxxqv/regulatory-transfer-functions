"""Evaluate every frozen locus against the causal-triangulation gates."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analyses/causal_triangulation/results"
LOCUS_ORDER = ["gata3", "stat3", "ptpn22"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def load_atac_peaks(path: Path) -> dict[str, list[tuple[int, int]]]:
    peaks: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            chromosome = fields[0].removeprefix("chr")
            peaks.setdefault(chromosome, []).append((int(fields[1]), int(fields[2])))
    return peaks


def nearest_peak(position: int, intervals: list[tuple[int, int]]) -> tuple[bool, int | None]:
    if not intervals:
        return False, None
    distances = [0 if start < position <= end else min(abs(position - start), abs(position - end)) for start, end in intervals]
    distance = min(distances)
    return distance == 0, int(distance)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "analyses/loci/results/locus_summary.csv")
    variants = pd.read_csv(ROOT / "analyses/loci/results/credible_set_variants.csv")
    features = pd.read_csv(ROOT / "analyses/loci/results/l2g_features.csv")
    programs = pd.read_csv(ROOT / "analyses/loci/results/transfer_programs.csv")
    pchic = pd.read_csv(ROOT / "analyses/loci/results/pchic_locus_summary.csv").set_index("locus")
    phenotypes = pd.read_parquet(ROOT / "analyses/primary/results/transfer_phenotypes.parquet")
    cis = pd.read_parquet(ROOT / "data/interim/eqtlgen/cis_mediators.parquet")
    trans = pd.read_parquet(ROOT / "data/interim/eqtlgen/trans_associations.parquet")
    atac_path = ROOT / "work/causal_inputs/ENCFF944LFH.bed.gz"
    peaks = load_atac_peaks(atac_path)

    feature_map: dict[tuple[str, str], float] = {}
    for row in features.itertuples(index=False):
        feature_map[(str(row.locus), str(row.feature))] = float(row.value) if pd.notna(row.value) else np.nan

    gate_records: list[dict[str, object]] = []
    instrument_records: list[dict[str, object]] = []
    direction_records: list[dict[str, object]] = []
    deletion_records: list[dict[str, object]] = []
    locus_records: list[dict[str, object]] = []

    for locus in LOCUS_ORDER:
        meta = summary.loc[summary["locus"] == locus].iloc[0]
        gene = str(meta["gene"])
        lead = str(meta["lead_rsid"])
        locus_variants = variants[(variants["locus"] == locus) & variants["is_95_credible_set"]]
        overlaps: list[str] = []
        distances: list[int] = []
        for variant in locus_variants.itertuples(index=False):
            overlap, distance = nearest_peak(
                int(variant.position), peaks.get(str(variant.chromosome).removeprefix("chr"), [])
            )
            if overlap:
                overlaps.append(str(variant.variant_id))
            if distance is not None:
                distances.append(distance)

        target_features = {
            name: feature_map.get((locus, name), np.nan)
            for name in ["eQtlColocH4Maximum", "eQtlColocClppMaximum", "e2gMean"]
        }
        coloc_value = target_features["eQtlColocH4Maximum"]
        coloc_pass = bool(np.isfinite(coloc_value) and coloc_value >= 0.80)
        e2g_value = target_features["e2gMean"]
        pchic_contact_variants = int(pchic.loc[locus, "variants_with_called_contact"])
        pchic_nearest = pchic.loc[locus, "nearest_called_contact_bp"]
        enhancer_pass = pchic_contact_variants > 0
        enhancer_evidence = (
            "Credible variant overlaps a called primary CD4 promoter contact"
            if enhancer_pass
            else "One rE2G feature and no credible-variant CD4 promoter contact do not meet the gate"
        )

        perturb = phenotypes[phenotypes["target_contrast_gene_name"] == gene].copy()
        cis_pass = bool(len(perturb) > 0 and perturb["ontarget_significant"].fillna(False).all())
        program = programs[programs["locus"] == locus].copy()
        program_pass = bool(len(program) > 0 and np.any(program["normalized_signed_effect"].fillna(0) != 0))

        lead_cis = cis[(cis["SNP"] == lead) & (cis["GeneSymbol"] == gene)].copy()
        lead_trans = trans[trans["SNP"] == lead].copy()
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
                    "valid_for_mediation": False,
                    "exclusion_reason": "fewer_than_three_independent_instruments",
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
            trans_gate_status = "passed" if direction_fraction > 0.5 and direction_p < 0.05 else "failed"
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
                            "deleted_direction_fraction": float(reduced["direction_match"].mean()) if len(reduced) else np.nan,
                            "estimate_change": float(reduced["direction_match"].mean() - full_fraction) if len(reduced) else np.nan,
                        }
                    )
        else:
            direction_p = np.nan
            direction_fraction = np.nan
            trans_gate_status = "unresolved"

        gates = [
            ("statistical_fine_mapping", "passed", float(meta["lead_posterior_probability"]), "Open Targets credible set"),
            ("gwas_cis_eqtl_colocalization", "passed" if coloc_pass else "failed", coloc_value, "Open Targets target-specific maximum H4"),
            ("credible_set_cd4_chromatin_overlap", "passed" if overlaps else "failed", len(overlaps), "ENCODE ENCSR841LHT IDR peaks"),
            ("enhancer_to_gene", "passed" if enhancer_pass else "failed", e2g_value, enhancer_evidence),
            ("perturbational_cis_effect", "passed" if cis_pass else "failed", float(perturb["ontarget_effect_size"].median()) if len(perturb) else np.nan, "Frozen CD4 on-target z-score"),
            ("signed_downstream_program", "passed" if program_pass else "failed", len(program), "Frozen signed CD4 program"),
            ("trans_eqtl_direction_concordance", trans_gate_status, direction_fraction, "eQTLGen lead-variant trans associations"),
        ]
        for gate, status, value, evidence in gates:
            gate_records.append(
                {"locus": locus, "gene": gene, "trait": meta["trait"], "gate": gate, "status": status, "value": value, "evidence": evidence}
            )

        passed = sum(status == "passed" for _, status, _, _ in gates)
        all_gates = passed == len(gates)
        mediation_status = "not_tested_assumptions_failed"
        if all_gates and len(lead_cis) >= 3:
            mediation_status = "eligible_not_estimated"
        tier = (
            "tier_2_convergent_without_valid_mediation"
            if all_gates
            else "tier_3_partial"
            if passed >= 4
            else "tier_4_unsupported_or_contradictory"
        )
        locus_records.append(
            {
                "locus": locus,
                "gene": gene,
                "trait": meta["trait"],
                "lead_rsid": lead,
                "credible_set_variants": len(locus_variants),
                "cd4_atac_overlap_variants": len(overlaps),
                "nearest_cd4_atac_peak_bp": min(distances) if distances else np.nan,
                "cd4_pchic_contact_variants": pchic_contact_variants,
                "nearest_cd4_pchic_contact_bp": pchic_nearest,
                "passed_gates": passed,
                "total_gates": len(gates),
                "evidence_tier": tier,
                "mediation_status": mediation_status,
                "mediation_claim": False,
                "primary_stop_reason": "not_all_evidence_gates_passed" if not all_gates else "inadequate_instruments",
            }
        )

    gates = pd.DataFrame(gate_records)
    directions = pd.DataFrame(direction_records)
    if len(directions):
        locus_tests = directions.groupby("locus")["direction_match"].agg(["sum", "count"]).reset_index()
        locus_tests["p_value"] = [binomtest(int(row["sum"]), int(row["count"]), 0.5, alternative="greater").pvalue for _, row in locus_tests.iterrows()]
        locus_tests["q_value"] = bh(locus_tests["p_value"])
    else:
        locus_tests = pd.DataFrame(columns=["locus", "sum", "count", "p_value", "q_value"])

    unavailable = pd.DataFrame(
        [
            {"analysis": "two_step_mediation", "locus": locus, "status": "not_tested", "reason": "colocalization or minimum instrument assumptions failed"}
            for locus in LOCUS_ORDER
        ]
        + [
            {"analysis": "matched_negative_locus", "locus": locus, "status": "unresolved", "reason": "MAF and local gene-density fields unavailable in the frozen source bundle"}
            for locus in LOCUS_ORDER
        ]
    )

    gates.to_csv(OUTPUT / "evidence_gates.csv", index=False)
    pd.DataFrame(locus_records).to_csv(OUTPUT / "locus_grades.csv", index=False)
    pd.DataFrame(instrument_records).to_csv(OUTPUT / "instrument_audit.csv", index=False)
    directions.to_csv(OUTPUT / "trans_direction_rows.csv", index=False)
    locus_tests.to_csv(OUTPUT / "trans_direction_tests.csv", index=False)
    pd.DataFrame(deletion_records).to_csv(OUTPUT / "leave_one_distal_gene.csv", index=False)
    unavailable.to_csv(OUTPUT / "unavailable_tests.csv", index=False)
    audit = {
        "eligible_locus_denominator": len(LOCUS_ORDER),
        "loci_with_mediation_claim": 0,
        "atac_source": "ENCODE ENCSR841LHT, ENCFF944LFH, GRCh38 IDR thresholded peaks",
        "atac_source_noncompliance_note": True,
        "atac_file_sha256": sha256(atac_path),
        "pchic_source": "Burren et al. 2017 Additional file 5 Table S4, GRCh37",
        "pchic_call_rule": "CHiCAGO score greater than 5 in either CD4 state",
        "all_loci_reported": True,
        "matched_negative_locus_status": "unresolved_missing_matching_fields",
    }
    (OUTPUT / "causal_triangulation_results.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
