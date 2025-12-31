import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

from analyses.independent_confirmation import acquire_genetic_inputs as acquisition


class MockHandler(BaseHTTPRequestHandler):
    files = {}
    modes = {}
    heads = 0
    gets = 0
    ranges = []
    if_matches = []
    truncated = set()
    get_etags = {}

    def log_message(self, *_):
        return

    def _payload(self):
        return self.files.get(self.path)

    def _headers(self, length, status=200, content_range=None, etag='"fixture-etag"'):
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        if content_range:
            self.send_header("Content-Range", content_range)
        self.end_headers()

    def do_HEAD(self):
        type(self).heads += 1
        payload = self._payload()
        if payload is None:
            self.send_error(404)
            return
        self._headers(len(payload))

    def do_GET(self):
        type(self).gets += 1
        payload = self._payload()
        if payload is None:
            self.send_error(404)
            return
        range_header = self.headers.get("Range", "")
        type(self).ranges.append(range_header)
        type(self).if_matches.append(self.headers.get("If-Match", ""))
        mode = self.modes.get(self.path, "valid")
        etag = self.get_etags.get(self.path, '"fixture-etag"')
        if range_header:
            start = int(range_header.removeprefix("bytes=").split("-", 1)[0])
            if mode == "ignored":
                self._headers(len(payload), 200, etag=etag)
                self.wfile.write(payload)
                return
            if mode == "invalid":
                body = payload[start:]
                self._headers(len(body), 206, f"bytes 0-{len(body) - 1}/{len(payload)}", etag)
                self.wfile.write(body)
                return
            body = payload[start:]
            self._headers(len(body), 206, f"bytes {start}-{len(payload) - 1}/{len(payload)}", etag)
            self.wfile.write(body)
            return
        self._headers(len(payload), 200, etag=etag)
        if mode == "truncate_once" and self.path not in self.truncated:
            type(self).truncated.add(self.path)
            self.wfile.write(payload[:4])
            self.wfile.flush()
            self.connection.shutdown(1)
            return
        self.wfile.write(payload)


class Fixture:
    fields = [
        "resource_id", "resource_type", "accession", "role", "selection_state",
        "data_url", "expected_bytes", "budget_bytes", "provider_checksum_algorithm",
        "provider_checksum", "opening_order", "outcome_opened", "notes",
    ]

    def __init__(self, base_url, payload=b"\x00locked\xffpayload"):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        (self.root / "analyses" / "independent_confirmation").mkdir(parents=True)
        self.data_root = self.root / "data" / "independent_confirmation" / "genetics"
        self.config = self.root / "config" / "genetic_confirmation_protocol.yaml"
        self.resources = self.root / "analyses" / "independent_confirmation" / "genetic_protocol_resources.tsv"
        self.payload = payload
        self.base_url = base_url
        self.rows = self._rows()
        self.protocol = self._protocol()
        self.write()

    def close(self):
        self.temp.cleanup()

    def _row(self, resource_id, resource_type, accession, role, path, opening="NA"):
        payload = MockHandler.files[path]
        return {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "accession": accession,
            "role": role,
            "selection_state": "selected",
            "data_url": self.base_url + path,
            "expected_bytes": str(len(payload)),
            "budget_bytes": str(len(payload)),
            "provider_checksum_algorithm": "MD5",
            "provider_checksum": hashlib.md5(payload).hexdigest(),
            "opening_order": str(opening),
            "outcome_opened": "false",
            "notes": "",
        }

    def _rows(self):
        program_relative = "analyses/program.bin"
        program_path = self.root / program_relative
        program_path.parent.mkdir(parents=True, exist_ok=True)
        program_path.write_bytes(b"frozen-program")
        program_hash = hashlib.sha256(program_path.read_bytes()).hexdigest()
        rows = [{
            "resource_id": "P01", "resource_type": "program_annotation", "accession": "PROGRAM",
            "role": "program_role", "selection_state": "selected_frozen", "data_url": program_relative,
            "expected_bytes": str(program_path.stat().st_size), "budget_bytes": "0",
            "provider_checksum_algorithm": "SHA256", "provider_checksum": program_hash,
            "opening_order": "0", "outcome_opened": "false", "notes": "",
        }]
        rows.append(self._row("B01", "baseline_ld", "BASE", "baseline_role", "/a.bin"))
        rows.append(self._row("R01", "build_reference", "BUILD", "build_role", "/build.bin"))
        rows.append(self._row("L01", "ld_reference", "LD", "ld_role", "/ld.bin"))
        types = ["negative_gwas_candidate"] * 2 + ["positive_gwas"] * 5 + ["eqtl"] * 6 + ["chromatin"] * 3
        for opening, resource_type in enumerate(types, start=1):
            rows.append(self._row(
                f"O{opening:02d}", resource_type, f"ACC{opening:02d}", f"role_{opening:02d}",
                f"/outcome_{opening:02d}.bin", opening,
            ))
        return rows

    def _protocol(self):
        program = self.rows[0]
        program_manifest = hashlib.sha256(
            f"{program['data_url']}\t{program['provider_checksum']}\n".encode("utf-8")
        ).hexdigest()
        return {
            "decision": "READY_FOR_CHECKSUM_DOWNLOAD",
            "sequential_checksum_download_ready": True,
            "outcome_opening_authorized": False,
            "outcome_files_opened_during_protocol_lock": 0,
            "checksums": {"redownload_attempts": 1},
            "program_annotation_freeze": {
                "freeze_before_disease_outcome": True,
                "manifest_sha256": program_manifest,
                "files": [{
                    "path": program["data_url"], "role": program["role"],
                    "bytes": int(program["expected_bytes"]), "sha256": program["provider_checksum"],
                }],
            },
            "dataset_roles": {
                "immutable_after_lock": True,
                "order": [
                    {"open": opening, "accession": f"ACC{opening:02d}", "role": f"role_{opening:02d}"}
                    for opening in range(1, 17)
                ],
            },
            "retention_and_capacity": {
                "reference_acquisition_order": acquisition.REFERENCE_ORDER,
                "chromosome_order": list(range(1, 23)),
                "locked_summary_budget_bytes": 100,
                "maximum_stage_temporary_bytes": 100,
                "tool_cache_budget_bytes": 100,
                "safety_margin_bytes": 100,
                "planned_peak_with_safety_bytes": sum(int(row["budget_bytes"]) for row in self.rows[1:]) + 400,
            },
        }

    def write(self):
        self.config.write_text(yaml.safe_dump(self.protocol, sort_keys=False), encoding="utf-8")
        with self.resources.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)

    @property
    def destination(self):
        return self.data_root / "001_B01" / "a.bin"

    def pin(self):
        acquisition.PINNED_PROTOCOL_SHA256 = hashlib.sha256(self.config.read_bytes()).hexdigest()
        acquisition.PINNED_RESOURCE_SHA256 = hashlib.sha256(self.resources.read_bytes()).hexdigest()
        acquisition.PINNED_REFERENCE_COUNTS = {"baseline": 1, "build": 1, "ld": 1}

    def acquirer(self, pin=True):
        if pin:
            self.pin()
        return acquisition.GeneticInputAcquirer(self.config, self.resources, self.data_root, timeout=5)


class GeneticAcquisitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payloads = {
            "/a.bin": b"\x00locked\xffpayload",
            "/build.bin": b"build",
            "/ld.bin": b"ld",
        }
        payloads.update({f"/outcome_{index:02d}.bin": bytes([index, 0, 255]) for index in range(1, 17)})
        MockHandler.files = payloads
        MockHandler.modes = {}
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        MockHandler.heads = 0
        MockHandler.gets = 0
        MockHandler.ranges = []
        MockHandler.if_matches = []
        MockHandler.truncated = set()
        MockHandler.get_etags = {}
        MockHandler.modes = {}
        self.fixture = Fixture(self.base_url)
        self.space = mock.patch.object(
            acquisition.shutil,
            "disk_usage",
            return_value=shutil._ntuple_diskusage(total=10000, used=0, free=10000),
        )
        self.space.start()

    def tearDown(self):
        self.space.stop()
        self.fixture.close()

    def run_first(self):
        return self.fixture.acquirer().run(max_files=1)

    def prepare_roster(self):
        self.assertEqual(self.fixture.acquirer().run(dry_run=True), acquisition.EXIT_OK)
        acquirer = self.fixture.acquirer()
        acquirer.load_and_validate()
        physicals = acquirer._load_roster()
        self.assertIsNotNone(physicals)
        return acquirer, physicals

    def test_fresh_download_and_manifest_hash(self):
        self.assertEqual(self.run_first(), acquisition.EXIT_OK)
        self.assertEqual(self.fixture.destination.read_bytes(), MockHandler.files["/a.bin"])
        manifest_path = self.fixture.data_root / "checksum_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["completion_state"], "partial")
        self.assertEqual(len(manifest["entries"]), 1)
        entry = manifest["entries"][0]
        self.assertEqual(entry["bytes"], len(MockHandler.files["/a.bin"]))
        self.assertEqual(entry["provider_checksum_status"], "verified")
        self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
        sidecar = manifest_path.with_suffix(".json.sha256").read_text(encoding="ascii").split()[0]
        self.assertEqual(sidecar, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        self.assertFalse(list(self.fixture.data_root.rglob("*.tmp")))

    def test_valid_resume(self):
        acquirer, physicals = self.prepare_roster()
        part = self.fixture.destination.with_name("a.bin.part")
        part.parent.mkdir(parents=True)
        part.write_bytes(MockHandler.files["/a.bin"][:4])
        acquirer._write_part_state(part, physicals[0], 4)
        MockHandler.heads = 0
        MockHandler.gets = 0
        MockHandler.ranges = []
        self.run_first()
        self.assertEqual(self.fixture.destination.read_bytes(), MockHandler.files["/a.bin"])
        self.assertIn("bytes=4-", MockHandler.ranges)

    def test_partial_state_binds_validator_and_frozen_identity(self):
        acquirer, physicals = self.prepare_roster()
        part = self.fixture.destination.with_name("a.bin.part")
        part.parent.mkdir(parents=True)
        part.write_bytes(MockHandler.files["/a.bin"][:4])
        acquirer._write_part_state(part, physicals[0], 4)
        state = json.loads(acquirer._part_state_path(part).read_text(encoding="utf-8"))
        self.assertEqual(state["url"], physicals[0].url)
        self.assertEqual(state["expected_bytes"], len(MockHandler.files["/a.bin"]))
        self.assertEqual(state["validator_kind"], "etag")
        self.assertEqual(state["validator"], '"fixture-etag"')
        self.assertEqual(state["protocol_sha256"], hashlib.sha256(self.fixture.config.read_bytes()).hexdigest())
        self.assertEqual(state["resource_table_sha256"], hashlib.sha256(self.fixture.resources.read_bytes()).hexdigest())

    def test_unbound_partial_restarts_without_range(self):
        self.prepare_roster()
        part = self.fixture.destination.with_name("a.bin.part")
        part.parent.mkdir(parents=True)
        part.write_bytes(MockHandler.files["/a.bin"][:4])
        MockHandler.heads = 0
        MockHandler.gets = 0
        MockHandler.ranges = []
        self.run_first()
        self.assertEqual(MockHandler.ranges, [""])
        self.assertEqual(self.fixture.destination.read_bytes(), MockHandler.files["/a.bin"])

    def test_complete_bound_part_finalizes_without_network(self):
        acquirer, physicals = self.prepare_roster()
        part = self.fixture.destination.with_name("a.bin.part")
        part.parent.mkdir(parents=True)
        part.write_bytes(MockHandler.files["/a.bin"])
        acquirer._write_part_state(part, physicals[0], len(MockHandler.files["/a.bin"]))
        MockHandler.heads = 0
        MockHandler.gets = 0
        self.run_first()
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)
        self.assertEqual(self.fixture.destination.read_bytes(), MockHandler.files["/a.bin"])
        self.assertFalse(acquirer._part_state_path(part).exists())

    def test_full_get_is_conditionally_bound(self):
        self.run_first()
        self.assertIn('"fixture-etag"', MockHandler.if_matches)

    def test_last_modified_is_used_when_etag_is_not_strong(self):
        metadata = acquisition.RemoteMetadata(
            url=self.base_url + "/a.bin",
            size=len(MockHandler.files["/a.bin"]),
            etag='W/"weak"',
            last_modified="Mon, 14 Sep 2026 00:00:00 GMT",
            accepts_ranges=True,
            checked_utc=acquisition.utc_now(),
        )
        self.assertEqual(
            self.fixture.acquirer()._conditional_headers(metadata),
            {"If-Unmodified-Since": "Mon, 14 Sep 2026 00:00:00 GMT"},
        )

    def test_changed_get_validator_is_rejected(self):
        self.prepare_roster()
        MockHandler.get_etags = {"/a.bin": '"changed-etag"'}
        MockHandler.heads = 0
        MockHandler.gets = 0
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_NETWORK)
        self.assertFalse(self.fixture.destination.exists())

    def test_full_get_may_omit_validator_with_locked_checksum(self):
        self.prepare_roster()
        MockHandler.get_etags = {"/a.bin": ""}
        self.run_first()
        self.assertEqual(self.fixture.destination.read_bytes(), MockHandler.files["/a.bin"])

    def test_full_get_may_not_omit_validator_without_locked_checksum(self):
        self.fixture.rows[1]["provider_checksum_algorithm"] = "NA"
        self.fixture.rows[1]["provider_checksum"] = "NA"
        self.fixture.write()
        self.prepare_roster()
        MockHandler.get_etags = {"/a.bin": ""}
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_NETWORK)
        self.assertFalse(self.fixture.destination.exists())

    def test_truncated_body_preserves_and_resumes_partial(self):
        self.prepare_roster()
        MockHandler.modes = {"/a.bin": "truncate_once"}
        MockHandler.heads = 0
        MockHandler.gets = 0
        MockHandler.ranges = []
        self.run_first()
        self.assertEqual(self.fixture.destination.read_bytes(), MockHandler.files["/a.bin"])
        self.assertEqual(MockHandler.gets, 2)
        self.assertIn("bytes=4-", MockHandler.ranges)

    def test_ignored_and_invalid_ranges_restart_safely(self):
        for mode in ["ignored", "invalid"]:
            with self.subTest(mode=mode):
                self.fixture.close()
                self.fixture = Fixture(self.base_url)
                MockHandler.gets = 0
                MockHandler.ranges = []
                MockHandler.modes = {"/a.bin": mode}
                acquirer, physicals = self.prepare_roster()
                MockHandler.gets = 0
                MockHandler.ranges = []
                part = self.fixture.destination.with_name("a.bin.part")
                part.parent.mkdir(parents=True)
                part.write_bytes(MockHandler.files["/a.bin"][:3])
                acquirer._write_part_state(part, physicals[0], 3)
                self.run_first()
                self.assertEqual(self.fixture.destination.read_bytes(), MockHandler.files["/a.bin"])
                self.assertEqual(MockHandler.gets, 2)
                self.assertEqual(MockHandler.ranges[0], "bytes=3-")
                self.assertEqual(MockHandler.ranges[1], "")

    def test_checksum_mismatch_stops_after_one_redownload(self):
        self.fixture.rows[1]["provider_checksum"] = "0" * 32
        self.fixture.write()
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_CHECKSUM)
        self.assertEqual(MockHandler.gets, 2)
        self.assertFalse(self.fixture.destination.exists())
        manifest = json.loads((self.fixture.data_root / "checksum_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["completion_state"], "stopped")

    def test_size_mismatch_stops_before_transfer(self):
        self.fixture.rows[1]["expected_bytes"] = str(len(MockHandler.files["/a.bin"]) + 1)
        self.fixture.rows[1]["budget_bytes"] = self.fixture.rows[1]["expected_bytes"]
        self.fixture.write()
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_SIZE)
        self.assertEqual(MockHandler.gets, 0)

    def test_insufficient_space_precedes_network(self):
        self.space.stop()
        with mock.patch.object(
            acquisition.shutil,
            "disk_usage",
            return_value=shutil._ntuple_diskusage(total=999, used=998, free=1),
        ):
            with self.assertRaises(acquisition.AcquisitionError) as caught:
                self.run_first()
        self.space.start()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_SPACE)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_restart_uses_remaining_plan_capacity(self):
        self.run_first()
        total = sum(len(payload) for payload in MockHandler.files.values())
        allowed = total - len(MockHandler.files["/a.bin"]) + 400
        self.space.stop()
        with mock.patch.object(
            acquisition.shutil,
            "disk_usage",
            return_value=shutil._ntuple_diskusage(total=allowed, used=0, free=allowed),
        ):
            self.assertEqual(self.fixture.acquirer().run(max_files=2), acquisition.EXIT_OK)
        self.space.start()
        self.assertTrue((self.fixture.data_root / "002_R01" / "build.bin").is_file())

    def test_live_external_consumption_stops_next_transfer(self):
        self.prepare_roster()
        high = sum(len(payload) for payload in MockHandler.files.values()) + 500
        usages = [
            shutil._ntuple_diskusage(total=high, used=0, free=high),
            shutil._ntuple_diskusage(total=high, used=0, free=high),
            shutil._ntuple_diskusage(total=high, used=high, free=0),
        ]
        self.space.stop()
        with mock.patch.object(acquisition.shutil, "disk_usage", side_effect=usages):
            with self.assertRaises(acquisition.AcquisitionError) as caught:
                self.fixture.acquirer().run(max_files=2)
        self.space.start()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_SPACE)
        self.assertTrue(self.fixture.destination.is_file())
        self.assertFalse((self.fixture.data_root / "002_R01" / "build.bin").exists())

    def test_role_order_refusal_precedes_network(self):
        self.fixture.protocol["dataset_roles"]["order"][0]["role"] = "changed_role"
        self.fixture.write()
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_ORDER)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_pinned_identity_refusal_precedes_network(self):
        acquisition.PINNED_PROTOCOL_SHA256 = "0" * 64
        acquisition.PINNED_RESOURCE_SHA256 = "0" * 64
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.fixture.acquirer(pin=False).run(max_files=1)
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PROTOCOL)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_opened_outcome_and_budget_overrun_are_refused(self):
        for field, value in [("outcome_opened", "true"), ("budget_bytes", "1")]:
            with self.subTest(field=field):
                self.fixture.close()
                self.fixture = Fixture(self.base_url)
                self.fixture.rows[4][field] = value
                self.fixture.write()
                with self.assertRaises(acquisition.AcquisitionError) as caught:
                    self.run_first()
                self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PROTOCOL)
                self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_unapproved_host_is_refused(self):
        self.fixture.rows[1]["data_url"] = "https://example.org/a.bin"
        self.fixture.write()
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PATH)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_unlisted_resource_refusal_precedes_network(self):
        extra = self.fixture._row("X01", "unexpected", "EXTRA", "extra_role", "/build.bin")
        self.fixture.rows.append(extra)
        self.fixture.write()
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_ORDER)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_idempotent_completed_file(self):
        self.run_first()
        first_gets = MockHandler.gets
        first_sha = hashlib.sha256(self.fixture.destination.read_bytes()).hexdigest()
        self.run_first()
        self.assertEqual(MockHandler.gets, first_gets)
        self.assertEqual(hashlib.sha256(self.fixture.destination.read_bytes()).hexdigest(), first_sha)
        manifest = json.loads((self.fixture.data_root / "checksum_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["entries"]), 1)

    def test_program_hash_mismatch_precedes_network(self):
        (self.fixture.root / "analyses" / "program.bin").write_bytes(b"changed-program")
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PROTOCOL)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_payload_content_is_not_logged(self):
        secret = MockHandler.files["/a.bin"].decode("latin1")
        logger = logging.getLogger("genetic_acquisition")
        with self.assertLogs(logger, level="INFO") as captured:
            self.run_first()
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_encoded_path_traversal_is_refused(self):
        self.fixture.rows[1]["data_url"] = self.base_url + "/%2e%2e/escape.bin"
        self.fixture.write()
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PATH)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_dry_run_and_manifest_only_do_not_download(self):
        self.assertEqual(self.fixture.acquirer().run(dry_run=True, max_files=1), acquisition.EXIT_OK)
        self.assertEqual(MockHandler.gets, 0)
        self.assertTrue((self.fixture.data_root / "acquisition_roster.json").is_file())
        self.assertFalse((self.fixture.data_root / "checksum_manifest.json").exists())
        self.run_first()
        self.assertEqual(self.fixture.acquirer().run(manifest_only=True), acquisition.EXIT_INCOMPLETE)
        self.assertEqual(MockHandler.gets, 1)
        manifest = json.loads((self.fixture.data_root / "checksum_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["entries"]), 19)

    def test_directory_resource_uses_only_locked_members(self):
        listing = b'<a href="keep.bin">keep</a><a href="extra.bin">extra</a>'
        MockHandler.files["/directory/"] = listing
        MockHandler.files["/directory/keep.bin"] = b"keep"
        MockHandler.files["/directory/extra.bin"] = b"extra"
        try:
            row = self.fixture.rows[1]
            row["data_url"] = self.base_url + "/directory/"
            row["expected_bytes"] = "4"
            row["budget_bytes"] = "4"
            row["provider_checksum_algorithm"] = "NA"
            row["provider_checksum"] = "NA"
            row["notes"] = "members=keep.bin;local_SHA256_required"
            self.fixture.write()
            self.assertEqual(self.fixture.acquirer().run(dry_run=True, max_files=1), acquisition.EXIT_OK)
            roster = json.loads((self.fixture.data_root / "acquisition_roster.json").read_text(encoding="utf-8"))
            first = [row for row in roster["physical_files"] if row["resource_id"] == "B01"]
            self.assertEqual([row["url"] for row in first], [self.base_url + "/directory/keep.bin"])
        finally:
            for path in ["/directory/", "/directory/keep.bin", "/directory/extra.bin"]:
                MockHandler.files.pop(path, None)

    def test_manifest_only_does_not_trust_prior_complete_state(self):
        self.prepare_roster()
        acquirer = self.fixture.acquirer()
        acquirer.load_and_validate()
        blank = {
            "schema_version": 1,
            "protocol_sha256": hashlib.sha256(self.fixture.config.read_bytes()).hexdigest(),
            "resource_table_sha256": hashlib.sha256(self.fixture.resources.read_bytes()).hexdigest(),
            "program_manifest_sha256": self.fixture.protocol["program_annotation_freeze"]["manifest_sha256"],
            "data_root": str(self.fixture.data_root),
            "generated_utc": acquisition.utc_now(),
            "completion_state": "complete",
            "entries": [],
        }
        acquisition.atomic_pair(
            acquirer.manifest_path,
            (json.dumps(blank, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        self.assertEqual(self.fixture.acquirer().run(manifest_only=True), acquisition.EXIT_INCOMPLETE)
        manifest = json.loads(acquirer.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["completion_state"], "incomplete")
        self.assertEqual(len(manifest["entries"]), 19)

    def test_manifest_only_requires_a_complete_local_roster(self):
        self.assertEqual(self.fixture.acquirer().run(manifest_only=True), acquisition.EXIT_INCOMPLETE)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)
        manifest = json.loads((self.fixture.data_root / "checksum_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["completion_state"], "incomplete")
        self.assertEqual(manifest["last_error"]["exit_code"], acquisition.EXIT_INCOMPLETE)

    def test_manifest_only_rejects_roster_denominator_change(self):
        self.prepare_roster()
        acquirer = self.fixture.acquirer()
        roster = json.loads(acquirer.roster_path.read_text(encoding="utf-8"))
        roster["physical_file_count"] -= 1
        acquisition.atomic_pair(
            acquirer.roster_path,
            (json.dumps(roster, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.fixture.acquirer().run(manifest_only=True)
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_ORDER)
        self.assertEqual(MockHandler.gets, 0)

    def test_atomic_pair_recovers_after_each_replacement(self):
        for stage in ["payload", "sidecar"]:
            with self.subTest(stage=stage):
                target = self.fixture.root / f"transaction_{stage}.json"
                with self.assertRaises(RuntimeError):
                    acquisition.atomic_pair(target, b'{"version":1}\n', interrupt_after=stage)
                acquisition.recover_atomic_pair(target)
                self.assertEqual(target.read_bytes(), b'{"version":1}\n')
                sidecar = target.with_suffix(".json.sha256")
                self.assertEqual(sidecar.read_text(encoding="ascii").split()[0], hashlib.sha256(target.read_bytes()).hexdigest())
                self.assertFalse(target.with_suffix(".json.txn").exists())

    def test_atomic_temp_collision_is_not_overwritten(self):
        target = self.fixture.root / "collision.json"
        collision = target.with_name(f".{target.name}.fixed.tmp")
        collision.write_bytes(b"sentinel")
        with mock.patch.object(acquisition.secrets, "token_hex", return_value="fixed"):
            with self.assertRaises(acquisition.AcquisitionError) as caught:
                acquisition.atomic_pair(target, b"new")
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PATH)
        self.assertEqual(collision.read_bytes(), b"sentinel")
        self.assertFalse(target.exists())

    def test_atomic_reparse_target_is_not_replaced(self):
        target = self.fixture.root / "protected.json"
        target.write_bytes(b"original")
        original = acquisition._is_reparse
        with mock.patch.object(
            acquisition,
            "_is_reparse",
            side_effect=lambda path: Path(path) == target or original(Path(path)),
        ):
            with self.assertRaises(acquisition.AcquisitionError) as caught:
                acquisition.atomic_pair(target, b"changed")
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PATH)
        self.assertEqual(target.read_bytes(), b"original")

    def test_concurrent_data_root_run_is_refused(self):
        with acquisition.DataRootLock(self.fixture.data_root):
            with self.assertRaises(acquisition.AcquisitionError) as caught:
                with acquisition.DataRootLock(self.fixture.data_root):
                    pass
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PATH)

    def test_reparse_data_root_is_refused(self):
        original = acquisition._is_reparse
        with mock.patch.object(
            acquisition,
            "_is_reparse",
            side_effect=lambda path: Path(path) == self.fixture.data_root or original(Path(path)),
        ):
            with self.assertRaises(acquisition.AcquisitionError) as caught:
                self.run_first()
        self.assertEqual(caught.exception.exit_code, acquisition.EXIT_PATH)
        self.assertEqual(MockHandler.heads + MockHandler.gets, 0)

    def test_source_contains_no_outcome_decoder(self):
        source = Path(acquisition.__file__).read_text(encoding="utf-8")
        for forbidden in ["import gzip", "import tarfile", "import zipfile", "import pandas", "import pyarrow", "read_csv(", "read_parquet("]:
            self.assertNotIn(forbidden, source)

    def test_owned_scope_and_ascii(self):
        root = Path(__file__).resolve().parents[1]
        owned = {
            "analyses/independent_confirmation/acquire_genetic_inputs.py",
            "analyses/independent_confirmation/README_acquisition.md",
            "tests/test_genetic_acquisition.py",
            ".gitignore",
        }
        subprocess = __import__("subprocess")
        commit = subprocess.run(
            ["git", "log", "--format=%H", "--grep=^add genetic acquisition$", "-1"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertTrue(commit)
        changed = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertTrue(set(changed).issubset(owned))
        for relative in owned:
            path = root / relative
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("\u2013", text)
                self.assertNotIn("\u2014", text)


if __name__ == "__main__":
    unittest.main()
