import csv
import hashlib
from pathlib import Path
import subprocess
import unittest
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "analyses" / "independent_confirmation" / "metadata" / "genetic_access_feasibility.tsv"
REPORT = ROOT / "analyses" / "independent_confirmation" / "metadata" / "genetic_access_feasibility.md"
OWNED = {
    "analyses/independent_confirmation/metadata/genetic_access_feasibility.tsv",
    "analyses/independent_confirmation/metadata/genetic_access_feasibility.md",
    "tests/test_genetic_access_feasibility.py",
}


def read_table():
    with TABLE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames, list(reader)


def row_hash(fields, row):
    included = [field for field in fields if field != "provisional_metadata_sha256"]
    payload = "".join(f"{field}={row[field]}\n" for field in included)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GeneticAccessFeasibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields, cls.rows = read_table()
        cls.by_id = {row["record_id"]: row for row in cls.rows}

    def test_complete_candidate_denominator(self):
        candidates = [row for row in self.rows if row["record_type"] == "candidate_gwas"]
        self.assertEqual(len(candidates), 5)
        self.assertEqual(
            {row["accession"] for row in candidates},
            {"GCST90475259", "GCST90627754", "GCST90475833", "GCST90475661", "GCST90476067"},
        )
        self.assertEqual(len({row["candidate_id"] for row in candidates}), 5)
        self.assertTrue(
            all(row["decision"] == "STOP_PENDING_PROTOCOL_AND_RESOURCE_LOCK" for row in candidates)
        )

    def test_all_shared_components_are_present(self):
        expected = {
            "ld_european_autosomes",
            "ld_european_sample_panel",
            "ld_pedigree_filter",
            "eqtl_cd4_unstimulated_sumstats",
            "eqtl_cd4_unstimulated_finemap",
            "eqtl_cd4_stim16_sumstats",
            "eqtl_cd4_stim16_finemap",
            "eqtl_cd4_stim40_sumstats",
            "eqtl_cd4_stim40_finemap",
            "eqtl_cd4_subtype_sensitivity",
            "atac_primary_cd4_states",
            "h3k27ac_primary_cd4_matrix",
            "h3k27ac_primary_cd4_peak_archive",
            "atac_build_harmonization",
            "negative_control_disease",
            "negative_control_loci",
            "partitioned_heritability_baseline",
        }
        self.assertEqual({row["role_id"] for row in self.rows if row["record_type"] != "candidate_gwas"}, expected)
        self.assertEqual(len(self.rows), 22)

    def test_exact_roles_are_unique_and_unlocked(self):
        roles = [row["role_id"] for row in self.rows]
        self.assertEqual(len(roles), len(set(roles)))
        for row in self.rows:
            self.assertEqual(
                row["role_lock_state"], "not_locked_pending_protocol_and_resource_capacity"
            )
            self.assertEqual(row["outcome_opening_order"], "not_assigned")
            self.assertEqual(row["decision"], "STOP_PENDING_PROTOCOL_AND_RESOURCE_LOCK")

    def test_no_outcome_artifacts_or_values(self):
        forbidden_columns = {
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
        self.assertFalse(forbidden_columns.intersection(self.fields))
        for row in self.rows:
            self.assertEqual(row["outcome_unopened"], "true")
            self.assertEqual(row["opened_in_audit"], "false")
            self.assertEqual(row["causal_gate_state"], "not_evaluated_pending_outcome_opening")
            self.assertTrue(row["final_sha256_state"].startswith("unavailable"))

    def test_metadata_hashes_are_deterministic(self):
        for row in self.rows:
            self.assertRegex(row["provisional_metadata_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(row["provisional_metadata_sha256"], row_hash(self.fields, row))

    def test_official_provenance(self):
        allowed = {
            "www.ebi.ac.uk",
            "ftp.ebi.ac.uk",
            "www.internationalgenome.org",
            "ftp.1000genomes.ebi.ac.uk",
            "www.ncbi.nlm.nih.gov",
            "ftp.ncbi.nlm.nih.gov",
            "hgdownload.soe.ucsc.edu",
            "genome.ucsc.edu",
            "creativecommons.org",
        }
        for row in self.rows:
            for field in ("official_metadata_url", "data_url_or_manifest", "terms_url"):
                value = row[field]
                if value == "NA":
                    continue
                self.assertEqual(urlparse(value).scheme, "https")
                self.assertIn(urlparse(value).hostname, allowed)

    def test_checksum_states_are_explicit(self):
        candidates = [row for row in self.rows if row["record_type"] == "candidate_gwas"]
        for row in candidates:
            self.assertRegex(row["provider_checksum"], r"^[0-9a-f]{32}$")
            self.assertEqual(row["checksum_algorithm"], "MD5")
            self.assertEqual(row["checksum_state"], "listed_not_verified")
        self.assertEqual(self.by_id["S01"]["provider_checksum"], "20220804_manifest.txt_has_44_MD5_values")
        no_checksum = [
            row
            for row in self.rows
            if row["checksum_state"] == "provider_checksum_unavailable_local_SHA256_required"
        ]
        self.assertEqual(len(no_checksum), 12)

    def test_resource_totals_and_build_boundaries(self):
        gwas = sum(int(row["file_size_bytes"]) for row in self.rows if row["record_type"] == "candidate_gwas")
        eqtl = sum(int(row["file_size_bytes"]) for row in self.rows if row["record_type"] == "shared_eqtl")
        exact_total = sum(int(row["file_size_bytes"]) for row in self.rows)
        self.assertEqual(gwas, 3696161913)
        self.assertEqual(eqtl, 9733743995)
        self.assertEqual(exact_total, 54624319078)
        self.assertEqual(self.by_id["S10"]["genome_build"], "hg19")
        self.assertEqual(self.by_id["S11"]["genome_build"], "GRCh38")
        self.assertEqual(self.by_id["S13"]["provider_checksum"], "35887f73fe5e2231656504d1f6430900")

    def test_frozen_blockers_prevent_premature_causal_claims(self):
        candidates = [row for row in self.rows if row["record_type"] == "candidate_gwas"]
        self.assertTrue(
            all(
                row["required_variant_qc_metadata_gate"] == "post_opening_schema_QC_pending"
                for row in candidates
            )
        )
        self.assertEqual(
            self.by_id["S14"]["negative_control_gate"],
            "pending_outcome_blind_negative_disease_lock",
        )
        self.assertEqual(
            self.by_id["S15"]["negative_control_gate"],
            "pending_outcome_blind_matching_rule_lock",
        )
        self.assertEqual(
            self.by_id["S16"]["results_gate"],
            "pending_exact_baseline_LD_release_lock",
        )
        self.assertFalse(any(row["causal_gate_state"] == "pass" for row in self.rows))

    def test_capacity_stop_and_gate_classes_are_reported(self):
        report = REPORT.read_text(encoding="utf-8")
        self.assertIn("STOP_PENDING_PROTOCOL_AND_RESOURCE_LOCK", report)
        self.assertIn("170.03 GiB", report)
        self.assertIn("220 to 300 GiB", report)
        self.assertIn("100 to 140 GiB", report)
        self.assertIn("resource-risk gate", report)
        self.assertIn("resolvable before outcome opening", report)
        self.assertIn("post-opening schema gate", report)
        self.assertNotIn("current hard execution blocker", report.lower())

    def test_no_disclosure_or_generation_metadata(self):
        forbidden = (
            "artificial " + "intelligence",
            "language " + "model",
            "generated " + "by",
            "ai " + "disclosure",
        )
        for path in (TABLE, REPORT, Path(__file__)):
            text = path.read_text(encoding="utf-8").lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, text)

    def test_no_unicode_dashes(self):
        for path in (TABLE, REPORT, Path(__file__)):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\u2013", text)
            self.assertNotIn("\u2014", text)

    def test_owned_file_boundary(self):
        result = subprocess.run(
            ["git", "log", "-n", "1", "--format=%H", "--grep=^assess genetic access$"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
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
