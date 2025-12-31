import csv
import hashlib
from pathlib import Path
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "genetic_confirmation_protocol.yaml"
REPORT = ROOT / "analyses" / "independent_confirmation" / "genetic_protocol_lock.md"
RESOURCES = ROOT / "analyses" / "independent_confirmation" / "genetic_protocol_resources.tsv"
OWNED = {
    "config/genetic_confirmation_protocol.yaml",
    "analyses/independent_confirmation/genetic_protocol_lock.md",
    "analyses/independent_confirmation/genetic_protocol_resources.tsv",
    "tests/test_genetic_protocol_lock.py",
}


def read_resources():
    with RESOURCES.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class GeneticProtocolLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cls.rows = read_resources()

    def test_decision_and_outcome_boundary(self):
        self.assertEqual(self.cfg["decision"], "READY_FOR_CHECKSUM_DOWNLOAD")
        self.assertTrue(self.cfg["sequential_checksum_download_ready"])
        self.assertNotIn("large_download_authorized", self.cfg)
        self.assertFalse(self.cfg["outcome_opening_authorized"])
        self.assertEqual(self.cfg["outcome_files_opened_during_protocol_lock"], 0)
        self.assertTrue(all(row["outcome_opened"] == "false" for row in self.rows))

    def test_positive_and_negative_denominators(self):
        positives = [row for row in self.rows if row["resource_type"] == "positive_gwas"]
        negatives = [row for row in self.rows if row["resource_type"] == "negative_gwas_candidate"]
        selected = [row for row in negatives if row["selection_state"] == "selected"]
        self.assertEqual(len(positives), 5)
        self.assertEqual(len(negatives), 5)
        self.assertEqual({row["accession"] for row in selected}, {"GCST90476060", "GCST90478363"})
        self.assertTrue(all(row["direct_phenotype"] == "true" for row in negatives))
        self.assertTrue(all(row["nonimmune_family"] in {"structural_hernia", "gastric_motility"} for row in negatives))

    def test_effective_sample_sizes_and_selection(self):
        by_accession = {row["accession"]: row for row in self.rows}
        for accession in ["GCST90476059", "GCST90476060", "GCST90476062", "GCST90476063", "GCST90478363"]:
            row = by_accession[accession]
            expected = 4 / (1 / int(row["cases"]) + 1 / int(row["controls"]))
            self.assertAlmostEqual(float(row["effective_n"]), expected, places=2)
        rule = self.cfg["negative_disease_selection"]
        self.assertEqual(rule["effective_n_formula"], "4/(1/cases+1/controls)")
        self.assertEqual(rule["allowed_range_ratio"], [0.90, 1.20])
        self.assertEqual(rule["allowed_effective_n_range"], [9882.23, 117148.39])
        self.assertEqual(rule["ranking"], "absolute_natural_log_ratio_to_family_anchor_ascending")
        self.assertEqual(rule["tie_break"], "accession_ascending")
        self.assertIn("family_history", rule["prohibited_phenotypes"])
        self.assertIn("biomarker", rule["prohibited_phenotypes"])

    def test_baseline_ld_is_exact(self):
        baseline = self.cfg["baseline_ld"]
        self.assertEqual(baseline["release"], "baseline-LD_v2.2")
        self.assertEqual(baseline["genome_build"], "GRCh37")
        self.assertEqual(baseline["weight_ld"], "1000G_Phase3_weights_hm3_no_MHC")
        self.assertEqual(baseline["frequency_files"], "1000G_Phase3_frq")
        for key in ["annotations_archive", "frequencies_archive", "plink_reference_archive", "weights_archive", "regression_snp_list", "version_record"]:
            self.assertRegex(baseline[key]["md5"], r"^[0-9a-f]{32}$")
            self.assertTrue(baseline[key]["url"].startswith("https://"))

    def test_numeric_region_calipers_are_complete(self):
        match = self.cfg["negative_region_matching"]
        self.assertEqual(match["exact_match"], ["chromosome"])
        self.assertEqual(match["controls_per_locus"], 10)
        self.assertEqual(match["minimum_controls_for_estimable_locus"], 5)
        self.assertTrue(match["no_caliper_expansion"])
        self.assertEqual(set(match["variables"]), {"maf", "locus_size", "ld_score", "gene_density", "distance_to_tss", "accessibility", "mappability"})
        for specification in match["variables"].values():
            numeric = [value for key, value in specification.items() if key in {"maximum", "minimum"}]
            self.assertTrue(numeric)
            self.assertTrue(all(isinstance(value, (int, float)) for value in numeric))
        self.assertTrue(match["association_values_forbidden_in_matching"])
        self.assertTrue(match["significance_values_forbidden_in_matching"])

    def test_liftover_rules_are_strict_and_counted(self):
        lift = self.cfg["liftover"]
        self.assertEqual(lift["input_coordinates"], "BED_0_based_half_open")
        self.assertEqual(lift["primary"]["min_match"], 0.95)
        self.assertEqual(lift["sensitivity"]["min_match"], 0.99)
        self.assertEqual(lift["multiple_mapping"], "reject")
        self.assertEqual(lift["split_mapping"], "reject")
        self.assertTrue(lift["source_blacklist_before_mapping"])
        self.assertTrue(lift["target_blacklist_after_mapping"])
        self.assertEqual(len(lift["denominators"]), 11)

    def test_roles_and_opening_order_are_unique_and_frozen(self):
        roles = self.cfg["dataset_roles"]
        ordered = roles["order"]
        self.assertTrue(roles["immutable_after_lock"])
        self.assertEqual([row["open"] for row in ordered], list(range(1, 17)))
        self.assertEqual(len({row["accession"] for row in ordered}), 16)
        self.assertEqual(len({row["role"] for row in ordered}), 16)
        self.assertEqual(ordered[0]["accession"], "GCST90478363")
        self.assertEqual(ordered[1]["accession"], "GCST90476060")
        self.assertFalse(roles["role_change_after_opening_allowed"])

    def test_program_annotations_precede_disease_outcomes(self):
        freeze = self.cfg["program_annotation_freeze"]
        self.assertTrue(freeze["freeze_before_disease_outcome"])
        self.assertFalse(freeze["rebuild_after_disease_outcome_allowed"])
        self.assertRegex(freeze["manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(freeze["files"]), 5)
        resource_rows = [row for row in self.rows if row["resource_type"] == "program_annotation"]
        self.assertEqual(len(resource_rows), 5)
        hashes = {row["data_url"]: row["provider_checksum"] for row in resource_rows}
        manifest_rows = []
        for file_spec in freeze["files"]:
            self.assertEqual(hashes[file_spec["path"]], file_spec["sha256"])
            path = ROOT / file_spec["path"]
            self.assertEqual(path.stat().st_size, file_spec["bytes"])
            observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed_hash, file_spec["sha256"])
            manifest_rows.append((file_spec["path"], observed_hash))
        payload = "".join(f"{path}\t{digest}\n" for path, digest in sorted(manifest_rows))
        self.assertEqual(hashlib.sha256(payload.encode("utf-8")).hexdigest(), freeze["manifest_sha256"])
        first_step = self.cfg["analysis_order"]["steps"][0]
        self.assertEqual(first_step, "verify_frozen_CD4_program_annotation_paths_hashes_and_manifest")

    def test_negative_controls_do_not_truncate_locked_diseases(self):
        rule = self.cfg["negative_disease_selection"]
        self.assertFalse(rule["negative_control_outcome_can_stop_later_disease_opening"])
        self.assertEqual(
            rule["disease_pipeline_stop_causes"],
            ["checksum_failure", "schema_failure", "unavailable_locked_resource"],
        )
        self.assertIn("every_schema_eligible_locked_disease", rule["locked_disease_continuation_rule"])
        self.assertEqual(self.cfg["locus_derivation"]["timing"], "after_each_schema_eligible_GWAS_is_opened")
        self.assertFalse(self.cfg["locus_derivation"]["outcome_magnitude_or_rank_used_for_matching"])

    def test_schema_stops_and_no_substitution(self):
        schema = self.cfg["post_opening_schema"]
        required = {"effect_allele", "other_allele", "standard_error", "p_value", "sample_size", "effect_allele_frequency_or_minor_allele_frequency"}
        self.assertTrue(required.issubset(schema["required_fields"]))
        self.assertEqual(schema["effect_field"], "exactly_one_of_beta_or_odds_ratio")
        self.assertEqual(schema["imputation_quality_minimum"], 0.80)
        self.assertGreaterEqual(len(schema["candidate_stop_rules"]), 7)
        self.assertGreaterEqual(len(schema["locus_stop_rules"]), 4)
        self.assertFalse(schema["substitutions_allowed"])

    def test_checksums_are_mandatory(self):
        checks = self.cfg["checksums"]
        self.assertTrue(checks["provider_checksum_required_when_available"])
        self.assertTrue(checks["provider_checksum_must_verify"])
        self.assertTrue(checks["local_sha256_required_for_every_file"])
        self.assertTrue(checks["checksum_manifest_sha256_required"])
        selected_with_md5 = [row for row in self.rows if row["selection_state"] == "selected" and row["provider_checksum_algorithm"] == "MD5"]
        self.assertTrue(selected_with_md5)
        self.assertTrue(all(len(row["provider_checksum"]) == 32 for row in selected_with_md5))

    def test_analysis_order_and_causal_gates_are_frozen(self):
        block = self.cfg["analysis_order"]
        order = block["steps"]
        self.assertGreater(order.index("execute_negative_diseases_through_complete_pipeline"), order.index("verify_frozen_CD4_program_annotation_paths_hashes_and_manifest"))
        self.assertLess(order.index("competitive_gene_set_enrichment"), order.index("stratified_LD_score_regression"))
        self.assertLess(order.index("locus_fine_mapping"), order.index("GWAS_cis_eQTL_colocalization"))
        self.assertLess(order.index("causal_instrument_gates"), order.index("mediation_only_for_passed_gates"))
        self.assertFalse(block["alternatives_after_outcome_inspection_allowed"])

    def test_capacity_arithmetic_and_retention(self):
        cap = self.cfg["retention_and_capacity"]
        source = sum(int(row["budget_bytes"]) for row in self.rows)
        self.assertEqual(source, cap["source_file_budget_bytes"])
        expected_peak = source + cap["locked_summary_budget_bytes"] + cap["maximum_stage_temporary_bytes"] + cap["tool_cache_budget_bytes"]
        self.assertEqual(expected_peak, cap["planned_peak_before_safety_bytes"])
        self.assertEqual(expected_peak + cap["safety_margin_bytes"], cap["planned_peak_with_safety_bytes"])
        self.assertEqual(cap["free_bytes"] - cap["planned_peak_with_safety_bytes"], cap["remaining_headroom_bytes"])
        self.assertTrue(cap["budget_fits"])
        self.assertTrue(cap["source_files_never_deleted"])
        self.assertFalse(cap["delete_source_files"])
        self.assertGreater(cap["safety_margin_bytes"], 0)

    def test_no_outcome_values_or_local_artifacts(self):
        forbidden_columns = {"observed_beta", "observed_odds_ratio", "observed_p_value", "observed_q_value", "posterior_probability", "heritability"}
        self.assertFalse(forbidden_columns.intersection(self.rows[0]))
        artifact_root = ROOT / "data" / "independent_confirmation" / "genetics"
        self.assertFalse(artifact_root.exists())

    def test_ascii_dashes_and_prohibited_metadata_language(self):
        prohibited = ("artificial " + "intelligence", "language " + "model", "generated " + "by", "ai " + "disclosure")
        for path in [CONFIG, REPORT, RESOURCES, Path(__file__)]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\u2013", text)
            self.assertNotIn("\u2014", text)
            for phrase in prohibited:
                self.assertNotIn(phrase, text.lower())

    def test_owned_file_boundary(self):
        commit = subprocess.run(
            ["git", "log", "-n", "1", "--format=%H", "--grep=^lock genetic protocol$"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if commit:
            changed = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
        else:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            changed = [line[3:].replace("\\", "/") for line in status if len(line) > 3]
        self.assertTrue(set(changed).issubset(OWNED))


if __name__ == "__main__":
    unittest.main()
