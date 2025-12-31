import csv
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "analyses" / "independent_confirmation" / "metadata"
CANDIDATES = METADATA / "genetic_candidates.tsv"
SOURCES = METADATA / "genetic_sources.tsv"
SCREEN = METADATA / "genetic_screen.md"


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames, list(reader)


def metadata_hash(fieldnames, row):
    fields = [field for field in fieldnames if field != "provisional_metadata_sha256"]
    value = "".join(f"{field}={row[field]}\n" for field in fields)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class GeneticMetadataScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate_fields, cls.candidates = read_tsv(CANDIDATES)
        cls.source_fields, cls.sources = read_tsv(SOURCES)

    def test_complete_disease_denominator(self):
        expected = {
            "ankylosing spondylitis",
            "asthma",
            "atopic eczema",
            "autoimmune disease",
            "celiac disease",
            "Crohn's disease",
            "Hashimoto's thyroiditis",
            "inflammatory bowel disease",
            "multiple sclerosis",
            "psoriasis",
            "rheumatoid arthritis",
            "systemic lupus erythematosus",
            "type 1 diabetes mellitus",
            "ulcerative colitis",
        }
        self.assertEqual(len(self.candidates), 14)
        self.assertEqual({row["disease"] for row in self.candidates}, expected)
        self.assertEqual(len({row["candidate_id"] for row in self.candidates}), 14)
        self.assertEqual(len({row["gwas_accession"] for row in self.candidates}), 14)

    def test_candidate_schema_has_all_opening_gates(self):
        required = {
            "trait_match_gate",
            "complete_gwas_gate",
            "ancestry_label_gate",
            "ancestry_matched_ld_gate",
            "immune_cell_eqtl_gate",
            "state_relevant_chromatin_gate",
            "fine_mapping_credible_set_status",
            "independent_molecular_convergence_gate",
            "mhc_primary_plan_gate",
            "negative_control_plan_gate",
            "novelty_gate",
            "results_gate",
            "provider_md5",
            "provider_md5_status",
            "provider_checksum_verification_gate",
            "final_full_file_sha256_gate",
            "checksum_lock_status",
            "outcome_opening_gate",
            "h27_opening_gate",
            "h28_colocalization_gate",
            "h28_instrument_strength_gate",
            "h28_directionality_gate",
            "h28_heterogeneity_gate",
            "h28_pleiotropy_gate",
            "h28_leave_one_instrument_gate",
            "h28_mediation_gate",
        }
        self.assertTrue(required.issubset(self.candidate_fields))
        for row in self.candidates:
            self.assertTrue(all(row[field] for field in required))

    def test_outcomes_and_final_checksums_remain_unopened(self):
        for row in self.candidates:
            self.assertEqual(row["outcomes_inspected_before_freeze"], "false")
            self.assertEqual(row["outcome_unopened"], "true")
            self.assertEqual(row["outcome_columns_opened"], "false")
            self.assertRegex(row["provider_md5"], r"^[0-9a-f]{32}$")
            self.assertEqual(row["provider_md5_status"], "listed_not_verified")
            self.assertEqual(row["provider_checksum_verification_gate"], "unavailable")
            self.assertEqual(row["final_full_file_sha256_gate"], "unavailable")
            self.assertEqual(row["checksum_lock_status"], "not_locked_outcome_unopened")
            self.assertEqual(row["outcome_opening_gate"], "unavailable")

        for row in self.sources:
            self.assertEqual(row["opened_in_this_screen"], "false")
            if row["outcomes_inspected_before_freeze"] == "false":
                self.assertEqual(row["outcome_unopened"], "true")

    def test_h27_metadata_selection_and_frozen_ranking(self):
        advances = [row for row in self.candidates if row["decision"] == "ADVANCE_TO_CHECKSUM_AND_ROLE_LOCK"]
        stops = [row for row in self.candidates if row["decision"] == "STOP"]
        unavailable = [row for row in self.candidates if row["decision"] == "UNAVAILABLE"]
        self.assertEqual(len(advances), 5)
        self.assertEqual(len(stops), 4)
        self.assertEqual(len(unavailable), 5)
        self.assertEqual(
            {row["disease"] for row in stops},
            {"autoimmune disease", "inflammatory bowel disease", "psoriasis", "rheumatoid arthritis"},
        )
        for row in stops:
            self.assertEqual(row["trait_match_gate"], "fail")
            self.assertEqual(row["h27_opening_gate"], "fail")
            self.assertEqual(row["selection_eligible"], "false")
            self.assertEqual(row["selection_rank"], "")

        for row in unavailable:
            self.assertEqual(row["trait_match_gate"], "pass")
            self.assertEqual(row["complete_gwas_gate"], "unavailable")
            self.assertEqual(row["h27_opening_gate"], "unavailable")
            self.assertEqual(row["selection_eligible"], "false")
            self.assertEqual(row["selection_rank"], "")

        for row in advances:
            for gate in (
                "trait_match_gate",
                "complete_gwas_gate",
                "ancestry_label_gate",
                "ancestry_matched_ld_gate",
                "immune_cell_eqtl_gate",
                "state_relevant_chromatin_gate",
                "independent_molecular_convergence_gate",
                "mhc_primary_plan_gate",
                "negative_control_plan_gate",
                "novelty_gate",
                "results_gate",
            ):
                self.assertEqual(row[gate], "pass")
            self.assertEqual(row["selection_eligible"], "true")
            self.assertEqual(row["h27_opening_gate"], "unavailable")

        ranked = sorted(advances, key=lambda row: int(row["selection_rank"]))
        self.assertEqual([int(row["selection_rank"]) for row in ranked], list(range(1, 6)))
        self.assertEqual(
            [row["gwas_accession"] for row in ranked],
            sorted(row["gwas_accession"] for row in advances),
        )

    def test_h28_and_mediation_are_unavailable(self):
        h28 = (
            "h28_colocalization_gate",
            "h28_instrument_strength_gate",
            "h28_directionality_gate",
            "h28_heterogeneity_gate",
            "h28_pleiotropy_gate",
            "h28_leave_one_instrument_gate",
            "h28_mediation_gate",
        )
        for row in self.candidates:
            self.assertTrue(all(row[field] == "unavailable" for field in h28))
            expected = (
                "derivable_after_opening"
                if row["complete_gwas_gate"] == "pass"
                else "unavailable_due_unverified_complete_statistics"
            )
            self.assertEqual(row["fine_mapping_credible_set_status"], expected)

    def test_direct_phenotypes_and_asthma_replacement(self):
        asthma = next(row for row in self.candidates if row["disease"] == "asthma")
        self.assertEqual(asthma["gwas_accession"], "GCST90475259")
        self.assertEqual(asthma["reported_trait"], "Asthma")
        self.assertEqual(asthma["phenotype_definition_type"], "direct_disease")
        self.assertEqual(asthma["trait_match_gate"], "pass")

        rejected = next(row for row in self.sources if row["accession"] == "GCST90476372")
        self.assertEqual(rejected["source_type"], "gwas_rejected_candidate")
        self.assertEqual(rejected["source_gate_status"], "fail")
        self.assertIn("family-history proxy", rejected["failure_reason"])

        failed_types = {row["phenotype_definition_type"] for row in self.candidates if row["trait_match_gate"] == "fail"}
        self.assertEqual(failed_types, {"biomarker_proxy", "broad_composite", "broad_phecode"})

    def test_full_summary_file_metadata_and_completeness(self):
        required = (
            "full_summary_statistics_url",
            "full_summary_statistics_metadata_url",
            "full_summary_statistics_file_name",
            "full_summary_statistics_file_format",
            "full_summary_statistics_build",
            "full_summary_statistics_file_size_bytes",
            "file_metadata_variant_count",
            "catalog_reported_variant_count",
            "provider_md5",
            "completeness_reason",
        )
        sparse = {
            "GCST90079532",
            "GCST90080337",
            "GCST90077860",
            "GCST90077870",
            "GCST90081487",
            "GCST90077810",
        }
        self.assertEqual(
            {row["gwas_accession"] for row in self.candidates if row["complete_gwas_gate"] == "unavailable"},
            sparse,
        )
        for row in self.candidates:
            self.assertTrue(all(row[field] for field in required))
            self.assertTrue(row["full_summary_statistics_url"].endswith(row["full_summary_statistics_file_name"]))
            self.assertEqual(
                row["full_summary_statistics_metadata_url"],
                row["full_summary_statistics_url"] + "-meta.yaml",
            )
            self.assertGreater(int(row["full_summary_statistics_file_size_bytes"]), 0)
            self.assertEqual(row["file_metadata_variant_count"], "unavailable")
            self.assertGreater(int(row["catalog_reported_variant_count"]), 0)
            if row["complete_gwas_gate"] == "pass":
                self.assertEqual(row["imputed"], "true")
                self.assertIn(row["full_summary_statistics_file_format"], {"GWAS-SSF v1.0", "pre-GWAS-SSF"})
            else:
                self.assertEqual(row["imputed"], "false")
                self.assertEqual(row["full_summary_statistics_file_format"], "pre-GWAS-SSF")

            if row["selection_eligible"] == "true":
                self.assertEqual(row["full_summary_statistics_file_format"], "GWAS-SSF v1.0")

        self.assertFalse(
            any(
                row["selection_eligible"] == "true" and row["complete_gwas_gate"] != "pass"
                for row in self.candidates
            )
        )

    def test_source_denominator_and_failures(self):
        self.assertEqual(len(self.sources), 28)
        self.assertEqual(len({row["source_id"] for row in self.sources}), 28)
        gwas = {row["accession"] for row in self.sources if row["source_type"] == "gwas_summary_statistics"}
        self.assertEqual(gwas, {row["gwas_accession"] for row in self.candidates})

        required = {
            "1000G_NYGC_30X",
            "QTS000040",
            "QTS000041",
            "GSE118189",
            "GSE244035",
            "EQTL_CATALOGUE_REST_API",
            "ENCSR841LHT",
            "ENCSR991PBP",
            "ENCSR561KOM",
            "GCST90652529",
            "FINNGEN_R12_K11_IBD_STRICT",
            "GCST000679",
            "GCST90476372",
        }
        self.assertTrue(required.issubset({row["accession"] for row in self.sources}))

        failed = {row["accession"] for row in self.sources if row["source_gate_status"] == "fail"}
        self.assertEqual(
            failed,
            {
                "GCST90012738",
                "GCST90081487",
                "GCST90476186",
                "GCST90476228",
                "GCST90476372",
                "ENCSR841LHT",
                "ENCSR991PBP",
                "ENCSR561KOM",
            },
        )
        unavailable = {row["accession"] for row in self.sources if row["source_gate_status"] == "unavailable"}
        self.assertEqual(
            unavailable,
            {
                "EQTL_CATALOGUE_REST_API",
                "GCST90079532",
                "GCST90080337",
                "GCST90077860",
                "GCST90077870",
                "GCST90077810",
            },
        )
        excluded = {row["accession"] for row in self.sources if row["source_gate_status"] == "excluded"}
        self.assertEqual(excluded, {"GCST90652529", "FINNGEN_R12_K11_IBD_STRICT", "GCST000679"})

    def test_source_metadata_are_complete(self):
        required = {
            "source_id",
            "source_type",
            "accession",
            "version",
            "provider",
            "official_url",
            "population_or_state",
            "genome_build",
            "data_availability",
            "fine_mapping_availability",
            "license",
            "license_url",
            "local_outcome_available",
            "full_summary_statistics_url",
            "full_summary_statistics_metadata_url",
            "full_summary_statistics_file_name",
            "full_summary_statistics_file_format",
            "full_summary_statistics_file_size_bytes",
            "file_metadata_variant_count",
            "catalog_reported_variant_count",
            "checksum_status",
            "outcomes_inspected_before_freeze",
            "outcome_unopened",
            "opened_in_this_screen",
            "source_gate_status",
            "metadata_retrieval_date",
            "hash_basis",
            "provisional_metadata_sha256",
        }
        self.assertTrue(required.issubset(self.source_fields))
        for row in self.sources:
            self.assertTrue(all(row[field] for field in required))
            self.assertTrue(row["official_url"].startswith("https://"))
            self.assertEqual(row["hash_basis"], "source_metadata_v2")

        gwas_rows = [row for row in self.sources if row["source_type"] == "gwas_summary_statistics"]
        candidate_by_accession = {row["gwas_accession"]: row for row in self.candidates}
        for row in gwas_rows:
            candidate = candidate_by_accession[row["accession"]]
            self.assertEqual(row["full_summary_statistics_url"], candidate["full_summary_statistics_url"])
            self.assertEqual(row["full_summary_statistics_file_size_bytes"], candidate["full_summary_statistics_file_size_bytes"])
            self.assertEqual(row["provider_checksum"], candidate["provider_md5"])
            self.assertEqual(row["checksum_status"], "provider_md5_listed_not_verified")

    def test_provisional_metadata_hashes_are_portable(self):
        for fields, rows in (
            (self.candidate_fields, self.candidates),
            (self.source_fields, self.sources),
        ):
            for row in rows:
                self.assertRegex(row["provisional_metadata_sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(row["provisional_metadata_sha256"], metadata_hash(fields, row))

    def test_no_outcome_value_columns_or_unicode_dashes(self):
        forbidden = {
            "p_value",
            "pvalue",
            "beta",
            "effect_size",
            "odds_ratio",
            "standard_error",
            "z_score",
            "q_value",
            "posterior_probability",
            "peak_signal",
        }
        self.assertFalse(forbidden.intersection(self.candidate_fields))
        self.assertFalse(forbidden.intersection(self.source_fields))
        for path in (CANDIDATES, SOURCES, SCREEN, Path(__file__)):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\u2013", text)
            self.assertNotIn("\u2014", text)


if __name__ == "__main__":
    unittest.main()
