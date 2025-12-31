"""Finalize the frozen metadata-only compositionality audit."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from analyses.compositionality.audit_metadata import ROOT, SOURCE, OUT, verify_inputs
from analyses.compositionality.screen_assignments import RESULTS

GEO = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
RAW = "https://raw.githubusercontent.com/sanderlab/scPerturb/master/"
ACCESSIONS = {
    "AdamsonWeissman2016": "GSE90546", "NormanWeissman2019": "GSE133344",
    "GasperiniShendure2019": "GSE120861", "TianKampmann2019": "GSE124703",
    "SunshineHein2023": "GSE208240", "YaoCleary2023": "GSE221321",
    "WesselsSatija2023": "GSE213957", "JoungZhang2023_combinatorial": "GSE217066",
    "JoungZhang2023_atlas": "GSE217460", "DixitRegev2016": "GSE90063",
    "Replogle2020_exp6": "GSE146194", "Pacalin2024_CRISPRi": "GSE220974",
    "Pacalin2024_CRISPRa": "GSE220974", "Cortical2025_double_knockdown": "GSE284197",
}


def accession(name):
    return next((value for key, value in ACCESSIONS.items() if name.startswith(key)), "not_resolved")


def audit_catalog():
    catalog = pd.read_csv(OUT / "scperturb_website.csv").fillna("")
    zenodo = json.loads((OUT / "scperturb_zenodo.json").read_text())
    released = {row["key"].removesuffix(".h5ad"): row for row in zenodo["files"]}
    denominators = {row["dataset"]: row for row in json.loads((RESULTS / "assignment_denominators.json").read_text())}
    aliases = {"NormanWeissman2019_filtered": "Norman2019", "GasperiniShendure2019_lowMOI": "Gasperini2019_lowMOI_TSS_only",
               "YaoCleary2023": "YaoCleary2023_KD_guide_pooled"}
    rows = []
    for _, source in catalog.iterrows():
        name = source["Full index"]
        intervention = source.Perturbation
        human = "Homo sapiens" in source.Organisms
        transcriptome = "RNA" in source.Modality and "(protein)" not in source.Modality
        compatible = intervention == "CRISPRi" or name == "YaoCleary2023"
        multiple = source.combined_perturbations == "y" or any(s in name for s in ["combinatorial", "YaoCleary"])
        item = released.get(name)
        report = denominators.get(aliases.get(name, name), {})
        reasons = []
        if not human:
            reasons.append("human_subset_unavailable")
        if not transcriptome:
            reasons.append("transcriptomic_readout_unavailable")
        if not compatible:
            reasons.append("nonconfirmatory_intervention")
        if not multiple:
            reasons.append("matching_double_target_design_not_established")
        if not item and name != "YaoCleary2023":
            reasons.append("absent_from_RNA_release_13350497")
        if report:
            for field, reason in [("pair_count_gate", "fewer_than_40_cell_eligible_pairs"),
                                  ("target_count_gate", "fewer_than_15_eligible_targets"),
                                  ("cd4_overlap_gate", "fewer_than_20_exact_CD4_targets")]:
                if not report[field]:
                    reasons.append(reason)
        elif compatible and multiple and transcriptome:
            reasons.append("exact_gene_pair_and_control_metadata_not_established")
        if "Adamson" in name and "10X005" in name:
            reasons.append("three_target_epistasis_design_below_target_and_pair_gates")
        if "Adamson" in name and "10X010" in name:
            reasons.append("harmonized_labels_are_guide_constructs_not_matched_gene_pairs")
        if "Gasperini" in name:
            reasons.append("enhancer_coperturbations_cannot_be_treated_as_gene_only_pairs")
        if "Schraivogel" in name:
            reasons.append("targeted_readout_overlap_and_gene_only_pairs_unverified")
        if "ReplogleWeissman2022" in name:
            reasons.append("same_gene_dual_guides_are_not_double_target_perturbations")
        if "Sunshine" in name:
            reasons.append("catalog_CRISPRi_conflicts_with_processing_CRISPR_cas9_label")
        if "TianKampmann2019" in name:
            reasons.append("harmonized_multiperturbation_assignments_not_independently_QC_validated")
        if name == "YaoCleary2023":
            reasons.append("KO_and_KD_not_pooled_for_inference")
        rows.append({"dataset": name, "screen_source": "scPerturb_metadata", "accession": accession(name),
                     "accession_or_doi": accession(name) if accession(name) != "not_resolved" else source.doi_url,
                     "organism": source.Organisms, "modality": source.Modality, "intervention": intervention,
                     "confirmatory_intervention": compatible, "combined_design_flag": source.combined_perturbations,
                     "catalog_cells": source["Total Number of Cells"], "catalog_conditions": source["Total Number of Perturbations"],
                     "processed_release_available": item is not None or name == "YaoCleary2023",
                     "release_bytes": item["size"] if item else None,
                     "provider_checksum": item["checksum"] if item else "not_available",
                     "provider_checksum_verified_on_full_release": False,
                     "license": "CC-BY-4.0" if item else "not_established",
                     "publication_url": source.doi_url, "metadata_url": RAW + "website/datavzrd/scperturb_dataset_info_datavzrd_annotated.csv",
                     "release_url": item["links"]["self"] if item else GEO + accession(name),
                     "cell_eligible_pairs": report.get("cell_eligible_pairs"), "eligible_targets": report.get("eligible_targets"),
                     "cd4_target_overlap": report.get("overlapping_cd4_targets"),
                     "both_components_cd4_pairs": report.get("both_components_cd4_pairs"),
                     "control_cells": report.get("controls"),
                     "guide_metadata": "assignment_metadata_retrieved" if report else "not_established",
                     "replicate_metadata": "technical_lanes_only" if name.startswith("Norman") else "batch_column_without_biological_unit_validation" if name.startswith("Tian") else "not_established",
                     "expressed_gene_overlap": "not_examined_after_earlier_gate_failure",
                     "selected_for_outcomes": False, "decision": "underpowered" if report and compatible else "unavailable",
                     "exclusion_reasons": ";".join(dict.fromkeys(reasons))})
    seen = set(catalog["Full index"])
    for name, item in released.items():
        if name in seen:
            continue
        is_dixit = name.startswith("Dixit")
        rows.append({"dataset": name, "screen_source": "Zenodo_release_only", "accession": accession(name),
                     "organism": "human" if is_dixit else "not_established", "intervention": "CRISPR-cas9" if is_dixit else "not_established",
                     "confirmatory_intervention": False if is_dixit else None, "processed_release_available": True,
                     "release_bytes": item["size"], "provider_checksum": item["checksum"],
                     "provider_checksum_verified_on_full_release": False, "license": "CC-BY-4.0",
                     "release_url": item["links"]["self"], "metadata_url": "https://zenodo.org/api/records/13350497",
                     "selected_for_outcomes": False, "decision": "unavailable",
                     "exclusion_reasons": "nonconfirmatory_intervention" if is_dixit else "missing_matching_design_metadata_in_screened_catalog"})
    for name in ["Replogle2020_exp6", "Pacalin2024_CRISPRi", "Pacalin2024_CRISPRa", "Cortical2025_double_knockdown"]:
        report = denominators.get(name, {})
        reasons = [reason for field, reason in [("pair_count_gate", "fewer_than_40_cell_eligible_pairs"),
                   ("target_count_gate", "fewer_than_15_eligible_targets"),
                   ("cd4_overlap_gate", "fewer_than_20_exact_CD4_targets")] if report and not report[field]]
        if name.endswith("CRISPRa"):
            reasons.append("nonconfirmatory_intervention")
        if name.startswith("Cortical"):
            reasons = ["two_target_ARX_LMO1_double_knockdown_below_target_and_pair_gates"]
        rows.append({"dataset": name, "screen_source": "original_repository", "accession": accession(name),
                     "organism": "Homo sapiens", "modality": "RNA", "intervention": "CRISPRa" if name.endswith("CRISPRa") else "CRISPRi",
                     "confirmatory_intervention": not name.endswith("CRISPRa"), "processed_release_available": True,
                     "license": "public_GEO_no_explicit_reuse_license", "metadata_url": GEO + accession(name),
                     "release_url": GEO + accession(name), "cell_eligible_pairs": report.get("cell_eligible_pairs"),
                     "eligible_targets": report.get("eligible_targets", 2 if name.startswith("Cortical") else None),
                     "cd4_target_overlap": report.get("overlapping_cd4_targets"),
                     "both_components_cd4_pairs": report.get("both_components_cd4_pairs"), "control_cells": report.get("controls"),
                     "guide_metadata": "assignment_metadata_retrieved" if report else "design_metadata_only",
                     "replicate_metadata": "two_batch_labels_independence_unverified" if name.startswith("Pacalin") else "technical_gemgroups" if name.startswith("Replogle") else "not_established",
                     "expressed_gene_overlap": "not_examined_after_earlier_gate_failure",
                     "selected_for_outcomes": False, "decision": "unavailable" if name.endswith("CRISPRa") else "underpowered", "exclusion_reasons": ";".join(reasons)})
    table = pd.DataFrame(rows)
    table["accession_or_doi"] = table.accession_or_doi.fillna(table.accession)
    table["full_eligibility_pass"] = False
    table["pair_model_power_gate"] = table.dataset.map(lambda name: denominators.get(aliases.get(name, name), {}).get("pair_model_power_gate"))
    table["no_outcome_data_accessed"] = True
    table["metadata_complete_for_selection"] = False
    table["selection_priority"] = table.confirmatory_intervention.eq(True).astype(int)
    table = table.sort_values(["selection_priority", "metadata_complete_for_selection", "cell_eligible_pairs", "cd4_target_overlap", "accession", "dataset"],
                              ascending=[False, False, False, False, True, True], na_position="last").reset_index(drop=True)
    table["screen_order"] = range(1, len(table) + 1)
    table.to_csv(RESULTS / "dataset_eligibility.tsv", sep="\t", index=False)
    return table


def provenance():
    original = {
        "replogle_exp6_cell_identities.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4367nnn/GSM4367984/suppl/GSM4367984_exp6.cell_identities.csv.gz",
        "norman_cell_identities.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_filtered_cell_identities.csv.gz",
        "pacalin_metadata.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE220nnn/GSE220974/suppl/GSE220974_K562_cell_metadata.csv.gz",
        "wessels_metadata.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213957/suppl/GSE213957_THP1-CaRPool-seq.metadata.tsv.gz",
        "yao_kd_perturbations.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6858nnn/GSM6858450/suppl/GSM6858450_KD_guide_pooled_perturbations.txt.gz",
        "gasperini_lowmoi_pheno.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120861/suppl/GSE120861_pilot_lowmoi_screen.phenoData.txt.gz",
        "gasperini_pilot_guide_groups.tsv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120861/suppl/GSE120861_grna_groups.pilot.txt.gz",
        "scperturb_website.csv": RAW + "website/datavzrd/scperturb_dataset_info_datavzrd_annotated.csv",
        "scperturb_zenodo.json": "https://zenodo.org/api/records/13350497",
        "geo_summaries.json": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id=200146194,200090546,200133344,200220974,200284197&retmode=json",
    }
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.name in original or path.name.endswith("_obs.tsv.gz") or path.name.endswith("_audit.json"):
            url = original.get(path.name, "https://zenodo.org/records/13350497")
            rows.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "url": url,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                         "contents": "assignment_and_design_metadata_only", "license": "public_GEO_no_explicit_reuse_license" if "ncbi" in url else "CC-BY-4.0_for_Zenodo_release" if "zenodo" in url else "repository_metadata_license_not_verified",
                         "retrieved_utc_date": "2026-09-13"})
    pd.DataFrame(rows).to_csv(RESULTS / "source_manifest.tsv", sep="\t", index=False)


def main():
    started = time.perf_counter()
    verified = verify_inputs()
    config = yaml.safe_load((SOURCE / "config/cell_systems_expansion.yaml").read_text())
    table = audit_catalog()
    provenance()
    decisions = pd.DataFrame([{"hypothesis": f"H{i}", "branch": "confirmatory_compositionality",
                               "question": config["compositionality"]["hypotheses"][f"H{i}"],
                               "eligible_datasets": 0, "outcome_pairs_analyzed": 0,
                               "effect_size": None, "ci_low": None, "ci_high": None, "p_value": None, "q_value": None,
                               "decision": "unavailable", "reason": "no_dataset_passed_all_frozen_availability_and_power_gates"}
                              for i in range(1, 7)])
    decisions.to_csv(RESULTS / "hypothesis_decisions.tsv", sep="\t", index=False)
    model_names = ["addition", "stronger_single", "mean_template", "signed_magnitude_cap", "tanh_saturation", "module_composition_30",
                   "covariate_ridge", "shallow_nonlinear", "network_propagation", "graph_model", "CPA", "GEARS"]
    pd.DataFrame([{"model": name, "fit_count": 0, "decision": "unavailable",
                   "reason": "dataset_gate_failed_before_model_input_contract_evaluation"} for name in model_names]).to_csv(RESULTS / "model_availability.tsv", sep="\t", index=False)
    pd.DataFrame([{"check": name, "performed": False, "decision": "unavailable", "reason": "no_eligible_outcome_branch"}
                  for name in ["residual_norm_direction_modules", "gain_buffering_rerouting", "emergent_lost_genes", "interaction_classes",
                               "pair_held_out", "leave_one_target", "guide_held_out", "dataset_held_out", "intervention_held_out",
                               "pair_bootstrap", "matched_pair_permutation", "sign_preserving_permutation", "random_modules",
                               "degree_preserving_rewiring", "NTC_response_controls", "same_target_response_controls", "edge_ablation",
                               "calibration_coverage", "BH_correction", "influence_diagnostics"]]).to_csv(RESULTS / "analysis_availability.tsv", sep="\t", index=False)
    summary = {"status": "stopped_at_metadata_gate", "catalog_records": 58, "release_files": 54,
               "screen_records_including_original_and_unmatched_releases": len(table),
               "verified_frozen_inputs": len(verified), "confirmatory_datasets_selected": 0,
               "external_expression_matrices_downloaded": 0, "external_expression_arrays_read": 0,
               "metadata_range_reads_note": "Remote HDF5 blocks were read only to extract obs metadata; no X, raw/X, layers or outcome summary arrays were accessed.",
               "models_fitted": 0, "norman_eligible_pairs": 127, "norman_both_cd4_pairs": 25,
               "norman_sensitivity_decision": "underpowered_for_CD4_composition_and_not_confirmatory",
               "norman_expression_downloaded": False, "thresholds_changed": False,
               "figure_S27": "not_created", "figure_reason": "Eligibility and model availability are fully reported in tables; no outcome result passed the gate and an extra figure would duplicate those tables.",
               "bootstrap_permutation_and_FDR": "not_run_no_estimable_outcome_hypothesis",
               "incidental_metadata_note": "The original Norman guide-design preview included an unused historical bulk-fitness column. This field was excluded from the audit and selection. No transcriptomic expression matrix or outcome array was opened.",
               "frozen_minimum_pairs": config["external_dataset_eligibility"]["confirmatory"]["minimum_unique_pairs"],
               "frozen_minimum_CD4_targets": config["external_dataset_eligibility"]["confirmatory"]["minimum_overlapping_cd4_targets"],
               "execution_seconds": time.perf_counter() - started}
    (RESULTS / "stopping_rule.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
