from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyses.independent_confirmation import run_genetic_confirmation as runner


HEADER = "chromosome\tbase_pair_location\teffect_allele\tother_allele\todds_ratio\tstandard_error\teffect_allele_frequency\tp_value\trsid\tn\tr2\n"


class GeneticConfirmationRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.output = self.root / "results"
        source = self.data / "017_N05/test.tsv.gz"
        source.parent.mkdir(parents=True)
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(HEADER)
            for chromosome in range(1, 23):
                handle.write(f"{chromosome}\t{1000 + chromosome}\tA\tG\t1.2\t0.1\t0.2\t1e-9\trs{chromosome}\t1000\t0.95\n")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.entry = {
            "resource_id": "N05",
            "relative_path": "017_N05/test.tsv.gz",
            "bytes": source.stat().st_size,
            "sha256": digest,
            "role": "negative_disease_low_neff",
        }
        self.manifest = {"entries": [self.entry]}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_stream_is_counted_and_written(self) -> None:
        with (
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "DATA", self.data),
            patch.object(runner, "OUTPUT", self.output),
            patch.object(runner, "EXPECTED_DENSE_VARIANTS", {"N05": 22}),
            patch.object(runner, "verify_acquisition", return_value=(self.manifest, "manifest")),
            patch.object(runner, "enforce_opening_order"),
        ):
            runner.scan_gwas("N05", 7)
        audit = json.loads((self.output / "schema_N05.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["counts"]["rows"], 22)
        self.assertEqual(audit["counts"]["harmonized_rows"], 22)
        self.assertEqual(audit["counts"]["significant_rows"], 22)
        self.assertTrue((self.output / "harmonized/N05.tsv.gz").exists())

    def test_missing_info_stops_before_rows(self) -> None:
        source = self.data / self.entry["relative_path"]
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(HEADER.replace("\tr2\n", "\n"))
        self.entry["bytes"] = source.stat().st_size
        self.entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        with (
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "DATA", self.data),
            patch.object(runner, "OUTPUT", self.output),
            patch.object(runner, "verify_acquisition", return_value=(self.manifest, "manifest")),
            patch.object(runner, "enforce_opening_order"),
        ):
            runner.scan_gwas("N05", 7)
        audit = json.loads((self.output / "schema_N05.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "failed")
        self.assertIn("imputation_quality_field_missing", audit["stop_reasons"])

    def test_cross_chunk_duplicates_are_exact(self) -> None:
        source = self.data / self.entry["relative_path"]
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(HEADER)
            for _ in range(3):
                handle.write("1\t1001\tA\tG\t1.2\t0.1\t0.2\t1e-9\trs1\t1000\t0.95\n")
            for chromosome in range(2, 23):
                handle.write(f"{chromosome}\t{1000 + chromosome}\tA\tG\t1.2\t0.1\t0.2\t1e-9\trs{chromosome}\t1000\t0.95\n")
        self.entry["bytes"] = source.stat().st_size
        self.entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        with (
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "DATA", self.data),
            patch.object(runner, "OUTPUT", self.output),
            patch.object(runner, "EXPECTED_DENSE_VARIANTS", {"N05": 24}),
            patch.object(runner, "REQUIRED_FREE_BYTES", 0),
            patch.object(runner, "verify_acquisition", return_value=(self.manifest, "manifest")),
            patch.object(runner, "enforce_opening_order"),
        ):
            runner.scan_gwas("N05", 2)
        audit = json.loads((self.output / "schema_N05.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["counts"]["duplicate_keys"], 2)
        self.assertIn("duplicate_fraction_exceeded", audit["stop_reasons"])

    def test_fractional_position_is_counted_not_cast(self) -> None:
        source = self.data / self.entry["relative_path"]
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(HEADER)
            handle.write("1\t1001.5\tA\tG\t1.2\t0.1\t0.2\t1e-9\trs_bad\t1000\t0.95\n")
            for chromosome in range(1, 23):
                handle.write(f"{chromosome}\t{2000 + chromosome}\tA\tG\t1.2\t0.1\t0.2\t1e-9\trs{chromosome}\t1000\t0.95\n")
        self.entry["bytes"] = source.stat().st_size
        self.entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        with (
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "DATA", self.data),
            patch.object(runner, "OUTPUT", self.output),
            patch.object(runner, "EXPECTED_DENSE_VARIANTS", {"N05": 23}),
            patch.object(runner, "verify_acquisition", return_value=(self.manifest, "manifest")),
            patch.object(runner, "enforce_opening_order"),
        ):
            runner.scan_gwas("N05", 5)
        audit = json.loads((self.output / "schema_N05.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["counts"]["canonical_coordinate_fail"], 1)

    def test_stage_lock_is_exclusive(self) -> None:
        with patch.object(runner, "OUTPUT", self.output):
            with runner.stage_lock("N05"):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with runner.stage_lock("N05"):
                        pass
            with runner.stage_lock("N05"):
                pass


if __name__ == "__main__":
    unittest.main()
