#!/usr/bin/env python3
import argparse
from contextlib import AbstractContextManager
import csv
from dataclasses import dataclass, replace
import errno
import hashlib
from html.parser import HTMLParser
import http.client
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import sys
from datetime import datetime, timezone
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

import yaml


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PROTOCOL = 10
EXIT_ORDER = 11
EXIT_SPACE = 12
EXIT_NETWORK = 13
EXIT_SIZE = 14
EXIT_CHECKSUM = 15
EXIT_PATH = 16
EXIT_INCOMPLETE = 17

REFERENCE_ORDER = [
    "baseline_LD",
    "genome_build_references",
    "chromosome_1_to_22_LD",
    "outcome_files_in_locked_order",
]
OUTCOME_TYPES = {"positive_gwas", "negative_gwas_candidate", "eqtl", "chromatin"}
BUILD_TYPES = {"build_reference", "matching_reference", "gene_annotation"}
LD_TYPES = {"ld_reference", "ld_metadata"}
NA_VALUES = {"", "NA", "N/A", "null", "None"}
CHUNK_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024
PINNED_PROTOCOL_SHA256 = "f2cd334f94dd156e6b9aabbaf8351343836f1a478634ac9a589c472315681318"
PINNED_RESOURCE_SHA256 = "b77069177245f5ca6fc87a66f835da7a07b9fa91bfa642a6da82ca2e8ce742d4"
PINNED_REFERENCE_COUNTS = {"baseline": 6, "build": 7, "ld": 3}
ALLOWED_HOSTS = {
    "zenodo.org",
    "hgdownload.soe.ucsc.edu",
    "raw.githubusercontent.com",
    "bismap.hoffmanlab.org",
    "ftp.ebi.ac.uk",
    "ftp.ncbi.nlm.nih.gov",
    "ftp.1000genomes.ebi.ac.uk",
}


class AcquisitionError(RuntimeError):
    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class RemoteMetadata:
    url: str
    size: int
    etag: str
    last_modified: str
    accepts_ranges: bool
    checked_utc: str


@dataclass(frozen=True)
class PhysicalFile:
    order_index: str
    resource_id: str
    accession: str
    role: str
    url: str
    relative_path: str
    expected_bytes: int
    provider_algorithm: str
    provider_checksum: str
    metadata: RemoteMetadata
    prefetched_bytes: bytes | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse(path: Path) -> bool:
    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = getattr(observed, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(observed.st_mode) or bool(attributes & reparse_flag)


def assert_safe_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    components = [absolute, *absolute.parents]
    for component in reversed(components):
        if component.exists() and _is_reparse(component):
            raise AcquisitionError(f"Reparse-point path component is forbidden: {component}", EXIT_PATH)


def ensure_safe_directory(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    missing = []
    probe = absolute
    while not probe.exists():
        missing.append(probe)
        probe = probe.parent
    assert_safe_components(probe)
    for component in reversed(missing):
        component.mkdir()
        if _is_reparse(component):
            raise AcquisitionError("Created directory became a reparse point", EXIT_PATH)


def _exclusive_write(path: Path, payload: bytes) -> None:
    assert_safe_components(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise AcquisitionError("Exclusive temporary path already exists", EXIT_PATH) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if path.exists() and not _is_reparse(path):
            path.unlink()
        raise


def _unique_temp(target: Path) -> Path:
    for _ in range(32):
        candidate = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
        if not candidate.exists():
            return candidate
    raise AcquisitionError("Cannot allocate an exclusive temporary path", EXIT_PATH)


def atomic_bytes(path: Path, payload: bytes) -> None:
    ensure_safe_directory(path.parent)
    assert_safe_components(path)
    temporary = _unique_temp(path)
    try:
        _exclusive_write(temporary, payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not _is_reparse(temporary):
            temporary.unlink()


def _pair_paths(path: Path) -> tuple[Path, Path]:
    return path.with_suffix(path.suffix + ".sha256"), path.with_suffix(path.suffix + ".txn")


def recover_atomic_pair(path: Path) -> None:
    sidecar, journal = _pair_paths(path)
    for protected in [path, sidecar, journal]:
        if protected.exists() and _is_reparse(protected):
            raise AcquisitionError("Transaction target cannot be a reparse point", EXIT_PATH)
    if not journal.exists():
        return
    if _is_reparse(journal):
        raise AcquisitionError("Transaction journal cannot be a reparse point", EXIT_PATH)
    try:
        transaction = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionError("Transaction journal is unreadable", EXIT_PROTOCOL) from error
    if transaction.get("version") != 1:
        raise AcquisitionError("Transaction journal version is unsupported", EXIT_PROTOCOL)
    payload_temp = safe_destination(path.parent, transaction.get("payload_temp", ""))
    sidecar_temp = safe_destination(path.parent, transaction.get("sidecar_temp", ""))
    expected = transaction.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise AcquisitionError("Transaction journal hash is invalid", EXIT_PROTOCOL)
    if not path.exists() or hash_file(path) != expected:
        if not payload_temp.is_file() or _is_reparse(payload_temp) or hash_file(payload_temp) != expected:
            raise AcquisitionError("Transaction payload cannot be recovered", EXIT_CHECKSUM)
        os.replace(payload_temp, path)
    expected_sidecar = f"{expected}  {path.name}\n".encode("ascii")
    if not sidecar.exists() or sidecar.read_bytes() != expected_sidecar:
        if sidecar_temp.is_file() and not _is_reparse(sidecar_temp) and sidecar_temp.read_bytes() == expected_sidecar:
            os.replace(sidecar_temp, sidecar)
        else:
            atomic_bytes(sidecar, expected_sidecar)
    for temporary in [payload_temp, sidecar_temp]:
        if temporary.exists() and not _is_reparse(temporary):
            temporary.unlink()
    journal.unlink()


def atomic_pair(path: Path, payload: bytes, interrupt_after: str | None = None) -> None:
    ensure_safe_directory(path.parent)
    for protected in [path, *_pair_paths(path)]:
        if protected.exists() and _is_reparse(protected):
            raise AcquisitionError("Transaction target cannot be a reparse point", EXIT_PATH)
    recover_atomic_pair(path)
    sidecar, journal = _pair_paths(path)
    payload_temp = _unique_temp(path)
    sidecar_temp = _unique_temp(sidecar)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    _exclusive_write(payload_temp, payload)
    _exclusive_write(sidecar_temp, sidecar_payload)
    transaction = {
        "version": 1,
        "payload_temp": payload_temp.name,
        "sidecar_temp": sidecar_temp.name,
        "sha256": digest,
    }
    atomic_bytes(journal, (json.dumps(transaction, sort_keys=True) + "\n").encode("utf-8"))
    os.replace(payload_temp, path)
    if interrupt_after == "payload":
        raise RuntimeError("simulated interruption after payload replacement")
    os.replace(sidecar_temp, sidecar)
    if interrupt_after == "sidecar":
        raise RuntimeError("simulated interruption after sidecar replacement")
    journal.unlink()


class DataRootLock(AbstractContextManager):
    def __init__(self, root: Path):
        self.root = root
        self.handle = None

    def __enter__(self):
        ensure_safe_directory(self.root)
        path = safe_destination(self.root, ".acquisition.lock")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        self.handle = os.fdopen(descriptor, "r+b", buffering=0)
        if _is_reparse(path):
            self.handle.close()
            self.handle = None
            raise AcquisitionError("Data-root lock path is a reparse point", EXIT_PATH)
        if path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self.handle.close()
            self.handle = None
            raise AcquisitionError("Another acquisition run holds the data-root lock", EXIT_PATH) from error
        return self

    def __exit__(self, *_):
        if self.handle is None:
            return False
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
        return False


def as_int(value: str | int | None) -> int | None:
    if value is None or str(value).strip() in NA_VALUES:
        return None
    try:
        parsed = int(str(value))
    except ValueError as error:
        raise AcquisitionError(f"Invalid byte count: {value}", EXIT_PROTOCOL) from error
    if parsed < 0:
        raise AcquisitionError(f"Negative byte count: {value}", EXIT_PROTOCOL)
    return parsed


def safe_url(url: str, allow_directory: bool = True) -> str:
    parsed = urlsplit(url)
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise AcquisitionError("Only HTTPS and loopback HTTP URLs are allowed", EXIT_PATH)
    if parsed.username or parsed.password or parsed.fragment:
        raise AcquisitionError("URL credentials and fragments are forbidden", EXIT_PATH)
    decoded = unquote(parsed.path).replace("\\", "/")
    if any(part == ".." for part in decoded.split("/")):
        raise AcquisitionError("URL path traversal is forbidden", EXIT_PATH)
    if not allow_directory and decoded.endswith("/"):
        raise AcquisitionError("A file URL is required", EXIT_PATH)
    if not loopback and parsed.hostname not in ALLOWED_HOSTS:
        raise AcquisitionError("URL host is not in the immutable acquisition set", EXIT_PATH)
    return url


def safe_destination(root: Path, relative_path: str) -> Path:
    root = Path(os.path.abspath(root))
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AcquisitionError("Destination path escapes the locked data root", EXIT_PATH)
    destination = Path(os.path.abspath(root / relative))
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise AcquisitionError("Destination path escapes the locked data root", EXIT_PATH) from error
    assert_safe_components(root)
    assert_safe_components(destination.parent)
    if destination.exists() and _is_reparse(destination):
        raise AcquisitionError("Destination cannot be a reparse point", EXIT_PATH)
    return destination


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


class GeneticInputAcquirer:
    def __init__(
        self,
        config_path: Path,
        resources_path: Path,
        data_root: Path,
        manifest_path: Path | None = None,
        timeout: float = 60.0,
    ):
        self.config_path = config_path.resolve()
        self.resources_path = resources_path.resolve()
        self.repo_root = self.config_path.parent.parent.resolve()
        self.data_root_input = Path(os.path.abspath(data_root))
        self.data_root = self.data_root_input
        self.manifest_path = Path(os.path.abspath(manifest_path or self.data_root / "checksum_manifest.json"))
        self.manifest_hash_path, self.manifest_journal_path = _pair_paths(self.manifest_path)
        self.roster_path = self.data_root / "acquisition_roster.json"
        self.roster_hash_path, self.roster_journal_path = _pair_paths(self.roster_path)
        self.timeout = timeout
        self.logger = logging.getLogger("genetic_acquisition")
        self.config: dict = {}
        self.rows: list[dict[str, str]] = []
        self.plan: list[dict[str, str]] = []
        self.manifest: dict = {}
        self.roster: dict = {}
        self.verified_urls: set[str] = set()
        self._loaded = False

    def load_and_validate(self) -> None:
        try:
            observed_protocol = hash_file(self.config_path)
            observed_resources = hash_file(self.resources_path)
        except OSError as error:
            raise AcquisitionError(f"Cannot hash the committed acquisition inputs: {error}", EXIT_PROTOCOL) from error
        if observed_protocol != PINNED_PROTOCOL_SHA256 or observed_resources != PINNED_RESOURCE_SHA256:
            raise AcquisitionError("Protocol or resource identity differs from the committed acquisition lock", EXIT_PROTOCOL)
        try:
            self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise AcquisitionError(f"Cannot read frozen protocol: {error}", EXIT_PROTOCOL) from error
        try:
            with self.resources_path.open(encoding="utf-8", newline="") as handle:
                self.rows = list(csv.DictReader(handle, delimiter="\t"))
        except OSError as error:
            raise AcquisitionError(f"Cannot read frozen resource table: {error}", EXIT_PROTOCOL) from error
        self._validate_protocol_state()
        self._verify_program_annotations()
        self.plan = self._build_plan()
        self._validate_data_root()
        self._loaded = True

    def _validate_protocol_state(self) -> None:
        if self.config.get("decision") != "READY_FOR_CHECKSUM_DOWNLOAD":
            raise AcquisitionError("Protocol is not ready for checksum download", EXIT_PROTOCOL)
        if not self.config.get("sequential_checksum_download_ready"):
            raise AcquisitionError("Sequential acquisition is not authorized", EXIT_PROTOCOL)
        if self.config.get("outcome_opening_authorized"):
            raise AcquisitionError("Acquisition cannot run under an outcome-opening protocol", EXIT_PROTOCOL)
        checks = self.config.get("checksums", {})
        if checks.get("redownload_attempts") != 1:
            raise AcquisitionError("The frozen single-redownload rule changed", EXIT_PROTOCOL)
        acquisition = self.config.get("retention_and_capacity", {}).get("reference_acquisition_order")
        if acquisition != REFERENCE_ORDER:
            raise AcquisitionError("Reference acquisition order changed", EXIT_ORDER)
        if self.config.get("dataset_roles", {}).get("immutable_after_lock") is not True:
            raise AcquisitionError("Dataset roles are not immutable", EXIT_ORDER)
        if self.config.get("outcome_files_opened_during_protocol_lock") != 0:
            raise AcquisitionError("The protocol records opened outcome files", EXIT_PROTOCOL)

    def _verify_program_annotations(self) -> None:
        freeze = self.config.get("program_annotation_freeze", {})
        specifications = freeze.get("files", [])
        if not specifications or not freeze.get("freeze_before_disease_outcome"):
            raise AcquisitionError("Program annotation freeze is incomplete", EXIT_PROTOCOL)
        program_rows = {
            row["data_url"]: row
            for row in self.rows
            if row.get("resource_type") == "program_annotation" and row.get("selection_state") == "selected_frozen"
        }
        manifest_rows: list[tuple[str, str]] = []
        for specification in specifications:
            relative = specification.get("path", "")
            path = safe_destination(self.repo_root, relative)
            if relative not in program_rows:
                raise AcquisitionError(f"Program resource is not listed: {relative}", EXIT_PROTOCOL)
            if not path.is_file() or path.is_symlink():
                raise AcquisitionError(f"Frozen program file is unavailable: {relative}", EXIT_PROTOCOL)
            if path.stat().st_size != int(specification["bytes"]):
                raise AcquisitionError(f"Frozen program byte count changed: {relative}", EXIT_PROTOCOL)
            observed = hash_file(path)
            if observed != specification["sha256"]:
                raise AcquisitionError(f"Frozen program hash changed: {relative}", EXIT_PROTOCOL)
            row = program_rows[relative]
            if row.get("provider_checksum") != observed or row.get("role") != specification.get("role"):
                raise AcquisitionError(f"Frozen program resource metadata changed: {relative}", EXIT_PROTOCOL)
            manifest_rows.append((relative, observed))
        payload = "".join(f"{path}\t{digest}\n" for path, digest in sorted(manifest_rows))
        observed_manifest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if observed_manifest != freeze.get("manifest_sha256"):
            raise AcquisitionError("Program annotation manifest hash changed", EXIT_PROTOCOL)

    def _build_plan(self) -> list[dict[str, str]]:
        if not self.rows:
            raise AcquisitionError("Resource table is empty", EXIT_PROTOCOL)
        required = {
            "resource_id", "resource_type", "accession", "role", "selection_state",
            "data_url", "expected_bytes", "budget_bytes", "provider_checksum_algorithm",
            "provider_checksum", "opening_order",
        }
        if not required.issubset(self.rows[0]):
            raise AcquisitionError("Resource table schema is incomplete", EXIT_PROTOCOL)
        ids = [row["resource_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise AcquisitionError("Resource IDs are not unique", EXIT_PROTOCOL)

        baseline = sorted(
            [row for row in self.rows if row["resource_type"] == "baseline_ld" and row["selection_state"] == "selected"],
            key=lambda row: natural_key(row["resource_id"]),
        )
        build = sorted(
            [row for row in self.rows if row["resource_type"] in BUILD_TYPES and row["selection_state"] == "selected"],
            key=lambda row: natural_key(row["resource_id"]),
        )
        ld = sorted(
            [row for row in self.rows if row["resource_type"] in LD_TYPES and row["selection_state"] == "selected"],
            key=lambda row: natural_key(row["resource_id"]),
        )
        opening_rows = [
            row for row in self.rows
            if row["resource_type"] in OUTCOME_TYPES and row["selection_state"] == "selected"
        ]
        if {
            "baseline": len(baseline),
            "build": len(build),
            "ld": len(ld),
        } != PINNED_REFERENCE_COUNTS:
            raise AcquisitionError("Reference resource denominator changed", EXIT_ORDER)
        try:
            opening_rows.sort(key=lambda row: int(row["opening_order"]))
        except ValueError as error:
            raise AcquisitionError("A selected outcome has no numeric opening order", EXIT_ORDER) from error

        locked_roles = self.config.get("dataset_roles", {}).get("order", [])
        if len(locked_roles) != 16 or [item.get("open") for item in locked_roles] != list(range(1, 17)):
            raise AcquisitionError("The frozen 16-role order is incomplete", EXIT_ORDER)
        if len(opening_rows) != 16:
            raise AcquisitionError("The resource table does not contain 16 selected outcome roles", EXIT_ORDER)
        for locked, row in zip(locked_roles, opening_rows):
            if (
                int(row["opening_order"]) != locked["open"]
                or row["accession"] != locked["accession"]
                or row["role"] != locked["role"]
            ):
                raise AcquisitionError("Resource table and frozen outcome-role order differ", EXIT_ORDER)
            if row.get("outcome_opened") != "false":
                raise AcquisitionError("A locked outcome is marked as opened", EXIT_PROTOCOL)

        recognized = {row["resource_id"] for row in baseline + build + ld + opening_rows}
        unlisted = [
            row["resource_id"]
            for row in self.rows
            if row["selection_state"] == "selected"
            and row["resource_type"] != "program_annotation"
            and row["resource_id"] not in recognized
        ]
        if unlisted:
            raise AcquisitionError(f"Selected resources have no frozen acquisition role: {','.join(unlisted)}", EXIT_ORDER)

        plan = baseline + build + ld + opening_rows
        urls: set[str] = set()
        for row in plan:
            url = row.get("data_url", "")
            safe_url(url)
            if url in urls:
                raise AcquisitionError("Duplicate acquisition URL in frozen resources", EXIT_ORDER)
            urls.add(url)
            if as_int(row.get("budget_bytes")) in {None, 0}:
                raise AcquisitionError(f"Selected resource has no byte budget: {row['resource_id']}", EXIT_PROTOCOL)
            expected = as_int(row.get("expected_bytes"))
            budget = as_int(row.get("budget_bytes"))
            if expected is not None and budget is not None and expected > budget:
                raise AcquisitionError(f"Expected bytes exceed budget for {row['resource_id']}", EXIT_PROTOCOL)
        return plan

    def _validate_data_root(self) -> None:
        if self.data_root == self.repo_root or self.data_root == Path(self.data_root.anchor):
            raise AcquisitionError("The data root is too broad", EXIT_PATH)
        assert_safe_components(self.data_root)
        try:
            manifest_relative = self.manifest_path.relative_to(self.data_root).as_posix()
        except ValueError as error:
            raise AcquisitionError("The checksum manifest must be inside the locked data root", EXIT_PATH) from error
        safe_destination(self.data_root, manifest_relative)
        if self.data_root.exists() and self.data_root.is_symlink():
            raise AcquisitionError("The locked data root cannot be a symlink", EXIT_PATH)

    def _load_manifest(self) -> dict:
        protocol_hash = hash_file(self.config_path)
        resources_hash = hash_file(self.resources_path)
        blank = {
            "schema_version": 1,
            "protocol_sha256": protocol_hash,
            "resource_table_sha256": resources_hash,
            "program_manifest_sha256": self.config["program_annotation_freeze"]["manifest_sha256"],
            "data_root": str(self.data_root),
            "generated_utc": utc_now(),
            "completion_state": "not_started",
            "entries": [],
        }
        recover_atomic_pair(self.manifest_path)
        if not self.manifest_path.exists():
            if self.manifest_hash_path.exists():
                raise AcquisitionError("Manifest hash exists without its manifest", EXIT_PROTOCOL)
            return blank
        if self.manifest_path.is_symlink() or not self.manifest_hash_path.is_file():
            raise AcquisitionError("Existing checksum manifest is not locked", EXIT_PROTOCOL)
        recorded = self.manifest_hash_path.read_text(encoding="ascii").strip().split()[0]
        observed = hash_file(self.manifest_path)
        if recorded != observed:
            raise AcquisitionError("Checksum manifest hash mismatch", EXIT_CHECKSUM)
        try:
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AcquisitionError("Checksum manifest is unreadable", EXIT_PROTOCOL) from error
        for key in ["protocol_sha256", "resource_table_sha256", "program_manifest_sha256", "data_root"]:
            if loaded.get(key) != blank[key]:
                raise AcquisitionError(f"Checksum manifest lock changed: {key}", EXIT_PROTOCOL)
        if not isinstance(loaded.get("entries"), list):
            raise AcquisitionError("Checksum manifest entries are invalid", EXIT_PROTOCOL)
        self._validate_manifest_entries(loaded["entries"])
        return loaded

    def _load_roster(self) -> list[PhysicalFile] | None:
        recover_atomic_pair(self.roster_path)
        if not self.roster_path.exists():
            if self.roster_hash_path.exists():
                raise AcquisitionError("Roster hash exists without its roster", EXIT_PROTOCOL)
            return None
        if _is_reparse(self.roster_path) or not self.roster_hash_path.is_file():
            raise AcquisitionError("Acquisition roster is not locked", EXIT_PROTOCOL)
        recorded = self.roster_hash_path.read_text(encoding="ascii").strip().split()[0]
        if recorded != hash_file(self.roster_path):
            raise AcquisitionError("Acquisition roster hash mismatch", EXIT_CHECKSUM)
        try:
            self.roster = json.loads(self.roster_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AcquisitionError("Acquisition roster is unreadable", EXIT_PROTOCOL) from error
        return self._validate_roster(self.roster)

    @staticmethod
    def _metadata_dict(metadata: RemoteMetadata) -> dict:
        return {
            "url": metadata.url,
            "size": metadata.size,
            "etag": metadata.etag,
            "last_modified": metadata.last_modified,
            "accepts_ranges": metadata.accepts_ranges,
            "checked_utc": metadata.checked_utc,
        }

    @classmethod
    def _physical_dict(cls, physical: PhysicalFile) -> dict:
        return {
            "order_index": physical.order_index,
            "resource_id": physical.resource_id,
            "accession": physical.accession,
            "role": physical.role,
            "url": physical.url,
            "relative_path": physical.relative_path,
            "expected_bytes": physical.expected_bytes,
            "provider_algorithm": physical.provider_algorithm,
            "provider_checksum": physical.provider_checksum,
            "metadata": cls._metadata_dict(physical.metadata),
        }

    @staticmethod
    def _physical_from_dict(item: dict) -> PhysicalFile:
        metadata = item.get("metadata", {})
        return PhysicalFile(
            order_index=item["order_index"],
            resource_id=item["resource_id"],
            accession=item["accession"],
            role=item["role"],
            url=item["url"],
            relative_path=item["relative_path"],
            expected_bytes=int(item["expected_bytes"]),
            provider_algorithm=item["provider_algorithm"],
            provider_checksum=item["provider_checksum"],
            metadata=RemoteMetadata(
                url=metadata["url"],
                size=int(metadata["size"]),
                etag=metadata.get("etag", ""),
                last_modified=metadata.get("last_modified", ""),
                accepts_ranges=bool(metadata.get("accepts_ranges")),
                checked_utc=metadata["checked_utc"],
            ),
        )

    def _url_allowed_for_row(self, url: str, row: dict[str, str]) -> bool:
        locked_url = row["data_url"]
        if locked_url.endswith("/"):
            return urlsplit(url).netloc == urlsplit(locked_url).netloc and url.startswith(locked_url)
        if row.get("provider_checksum_algorithm") == "manifest_MD5_set":
            prefix = locked_url.rsplit("/", 1)[0] + "/"
            return url == locked_url or (
                urlsplit(url).netloc == urlsplit(prefix).netloc and url.startswith(prefix)
            )
        return url == locked_url

    def _validate_roster(self, roster: dict) -> list[PhysicalFile]:
        identities = {
            "protocol_sha256": hash_file(self.config_path),
            "resource_table_sha256": hash_file(self.resources_path),
            "program_manifest_sha256": self.config["program_annotation_freeze"]["manifest_sha256"],
        }
        if roster.get("schema_version") != 1 or any(roster.get(key) != value for key, value in identities.items()):
            raise AcquisitionError("Acquisition roster identity changed", EXIT_PROTOCOL)
        logicals = roster.get("logical_resources")
        items = roster.get("physical_files")
        if not isinstance(logicals, list) or not isinstance(items, list) or len(logicals) != len(self.plan):
            raise AcquisitionError("Acquisition roster denominator changed", EXIT_ORDER)
        physicals: list[PhysicalFile] = []
        seen_urls: set[str] = set()
        for logical_index, (logical, row) in enumerate(zip(logicals, self.plan), start=1):
            expected_logical = {
                "logical_index": logical_index,
                "resource_id": row["resource_id"],
                "accession": row["accession"],
                "role": row["role"],
                "locked_url": row["data_url"],
            }
            if any(logical.get(key) != value for key, value in expected_logical.items()):
                raise AcquisitionError("Acquisition roster role or order changed", EXIT_ORDER)
            selected = [item for item in items if item.get("resource_id") == row["resource_id"]]
            if len(selected) != logical.get("physical_file_count") or not selected:
                raise AcquisitionError("Acquisition roster physical denominator changed", EXIT_ORDER)
            selected.sort(key=lambda item: tuple(int(part) for part in item["order_index"].split(".")))
            current = []
            for item in selected:
                try:
                    physical = self._physical_from_dict(item)
                except (KeyError, TypeError, ValueError) as error:
                    raise AcquisitionError("Acquisition roster physical record is invalid", EXIT_PROTOCOL) from error
                safe_url(physical.url, allow_directory=False)
                if physical.url in seen_urls or not self._url_allowed_for_row(physical.url, row):
                    raise AcquisitionError("Acquisition roster contains an unlisted URL", EXIT_ORDER)
                if physical.role != row["role"] or physical.accession != row["accession"]:
                    raise AcquisitionError("Acquisition roster contains an unlisted role", EXIT_ORDER)
                if physical.expected_bytes < 1 or physical.metadata.size != physical.expected_bytes:
                    raise AcquisitionError("Acquisition roster byte count is invalid", EXIT_SIZE)
                safe_destination(self.data_root, physical.relative_path)
                seen_urls.add(physical.url)
                current.append(physical)
                physicals.append(physical)
            aggregate = sum(
                physical.expected_bytes for physical in current if not physical.order_index.endswith(".000")
            )
            locked_expected = as_int(row.get("expected_bytes"))
            if locked_expected is not None and aggregate != locked_expected:
                raise AcquisitionError("Acquisition roster aggregate byte count changed", EXIT_SIZE)
            if aggregate > int(row["budget_bytes"]):
                raise AcquisitionError("Acquisition roster exceeds the locked budget", EXIT_SIZE)
        ordered = sorted(physicals, key=lambda item: tuple(int(part) for part in item.order_index.split(".")))
        if physicals != ordered or int(roster.get("physical_file_count", -1)) != len(physicals):
            raise AcquisitionError("Acquisition roster physical order changed", EXIT_ORDER)
        return physicals

    def _build_roster(self) -> list[PhysicalFile]:
        physicals: list[PhysicalFile] = []
        logicals = []
        for logical_index, row in enumerate(self.plan, start=1):
            expanded = self._expand(row, logical_index)
            physicals.extend(expanded)
            logicals.append({
                "logical_index": logical_index,
                "resource_id": row["resource_id"],
                "accession": row["accession"],
                "role": row["role"],
                "locked_url": row["data_url"],
                "physical_file_count": len(expanded),
            })
        self.roster = {
            "schema_version": 1,
            "protocol_sha256": hash_file(self.config_path),
            "resource_table_sha256": hash_file(self.resources_path),
            "program_manifest_sha256": self.config["program_annotation_freeze"]["manifest_sha256"],
            "generated_utc": utc_now(),
            "logical_resources": logicals,
            "physical_file_count": len(physicals),
            "physical_files": [self._physical_dict(physical) for physical in physicals],
        }
        self._validate_roster(self.roster)
        payload = (json.dumps(self.roster, indent=2, sort_keys=True) + "\n").encode("utf-8")
        atomic_pair(self.roster_path, payload)
        return physicals

    def _validate_manifest_entries(self, entries: list[dict]) -> None:
        by_id = {row["resource_id"]: row for row in self.plan}
        seen_urls: set[str] = set()
        for entry in entries:
            resource_id = entry.get("resource_id")
            row = by_id.get(resource_id)
            if row is None or entry.get("role") != row["role"]:
                raise AcquisitionError("Checksum manifest contains an unlisted role", EXIT_ORDER)
            url = safe_url(entry.get("url", ""), allow_directory=False)
            if url in seen_urls:
                raise AcquisitionError("Checksum manifest contains a duplicate URL", EXIT_ORDER)
            seen_urls.add(url)
            if not self._url_allowed_for_row(url, row):
                raise AcquisitionError("Checksum manifest contains an unlisted URL", EXIT_ORDER)
            safe_destination(self.data_root, entry.get("relative_path", ""))
            if entry.get("completion_state") not in {"complete", "missing", "stopped"}:
                raise AcquisitionError("Checksum manifest completion state is invalid", EXIT_PROTOCOL)

    def _write_manifest(self, state: str, last_error: AcquisitionError | None = None) -> None:
        self.manifest["generated_utc"] = utc_now()
        self.manifest["completion_state"] = state
        if last_error:
            self.manifest["last_error"] = {"exit_code": last_error.exit_code, "message": str(last_error)}
        else:
            self.manifest.pop("last_error", None)
        payload = (json.dumps(self.manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        atomic_pair(self.manifest_path, payload)

    def _fixed_overhead(self) -> int:
        capacity = self.config["retention_and_capacity"]
        return sum(int(capacity[key]) for key in [
            "locked_summary_budget_bytes",
            "maximum_stage_temporary_bytes",
            "tool_cache_budget_bytes",
            "safety_margin_bytes",
        ])

    def _disk_gate(self, remaining_source_bytes: int) -> None:
        required = remaining_source_bytes + self._fixed_overhead()
        probe = self.data_root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        if free < required:
            raise AcquisitionError(f"Insufficient free space: {free} bytes available, {required} required", EXIT_SPACE)

    def _request(self, url: str, method: str = "GET", headers: dict[str, str] | None = None):
        safe_url(url)
        request_headers = {"Accept-Encoding": "identity", "User-Agent": "regulatory-transfer-acquisition/1.0"}
        request_headers.update(headers or {})
        request = Request(url, method=method, headers=request_headers)
        try:
            response = urlopen(request, timeout=self.timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise AcquisitionError(f"Network metadata or transfer failed for a locked resource: {error}", EXIT_NETWORK) from error
        if response.geturl() != url:
            response.close()
            raise AcquisitionError("Redirected URLs are not in the frozen resource table", EXIT_PATH)
        if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
            response.close()
            raise AcquisitionError("Encoded transfer would change locked bytes", EXIT_NETWORK)
        return response

    def _remote_metadata(self, url: str) -> RemoteMetadata:
        checked = utc_now()
        try:
            response = self._request(url, method="HEAD")
        except AcquisitionError as error:
            if "HTTP Error 405" not in str(error) and "HTTP Error 501" not in str(error):
                raise
            response = self._request(url, headers={"Range": "bytes=0-0"})
        with response:
            status = getattr(response, "status", response.getcode())
            if status not in {200, 206}:
                raise AcquisitionError("Remote metadata request returned an invalid status", EXIT_NETWORK)
            length = response.headers.get("Content-Length")
            content_range = response.headers.get("Content-Range", "")
            if status == 206 and "/" in content_range:
                length = content_range.rsplit("/", 1)[1]
            try:
                size = int(length or "")
            except ValueError as error:
                raise AcquisitionError("Remote byte count is unavailable", EXIT_SIZE) from error
            if size < 0:
                raise AcquisitionError("Remote byte count is invalid", EXIT_SIZE)
            return RemoteMetadata(
                url=url,
                size=size,
                etag=response.headers.get("ETag", "").strip(),
                last_modified=response.headers.get("Last-Modified", ""),
                accepts_ranges=response.headers.get("Accept-Ranges", "").lower() == "bytes",
                checked_utc=checked,
            )

    def _read_metadata_bytes(self, url: str) -> tuple[bytes, RemoteMetadata]:
        try:
            with self._request(url) as response:
                status = getattr(response, "status", response.getcode())
                if status != 200:
                    raise AcquisitionError("Metadata file did not return a complete response", EXIT_NETWORK)
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_METADATA_BYTES:
                    raise AcquisitionError("Metadata response exceeds the fixed safety limit", EXIT_SIZE)
                payload = response.read(MAX_METADATA_BYTES + 1)
                if len(payload) > MAX_METADATA_BYTES:
                    raise AcquisitionError("Metadata response exceeds the fixed safety limit", EXIT_SIZE)
                if declared and len(payload) != int(declared):
                    raise AcquisitionError("Metadata response byte count changed", EXIT_SIZE)
                metadata = RemoteMetadata(
                    url=url,
                    size=len(payload),
                    etag=response.headers.get("ETag", "").strip(),
                    last_modified=response.headers.get("Last-Modified", ""),
                    accepts_ranges=response.headers.get("Accept-Ranges", "").lower() == "bytes",
                    checked_utc=utc_now(),
                )
                return payload, metadata
        except AcquisitionError:
            raise
        except (OSError, HTTPError, URLError, TimeoutError, http.client.HTTPException) as error:
            raise AcquisitionError(f"Metadata body read failed: {error}", EXIT_NETWORK) from error

    def _direct_physical(self, row: dict[str, str], logical_index: int) -> list[PhysicalFile]:
        url = safe_url(row["data_url"], allow_directory=False)
        metadata = self._remote_metadata(url)
        expected = as_int(row.get("expected_bytes"))
        budget = as_int(row.get("budget_bytes"))
        if expected is not None and metadata.size != expected:
            raise AcquisitionError(f"Remote byte count changed for {row['resource_id']}", EXIT_SIZE)
        if expected is None:
            if budget is None or metadata.size > budget:
                raise AcquisitionError(f"Remote file exceeds the locked budget for {row['resource_id']}", EXIT_SIZE)
            expected = metadata.size
        filename = self._filename_from_url(url)
        relative = f"{logical_index:03d}_{row['resource_id']}/{filename}"
        physical = PhysicalFile(
            order_index=f"{logical_index:03d}.001",
            resource_id=row["resource_id"],
            accession=row["accession"],
            role=row["role"],
            url=url,
            relative_path=relative,
            expected_bytes=expected,
            provider_algorithm=row.get("provider_checksum_algorithm", "NA"),
            provider_checksum=row.get("provider_checksum", "NA"),
            metadata=metadata,
        )
        self._verify_etag_lock(physical)
        return [physical]

    def _directory_physical(self, row: dict[str, str], logical_index: int) -> list[PhysicalFile]:
        base = safe_url(row["data_url"])
        payload, _ = self._read_metadata_bytes(base)
        parser = LinkParser()
        try:
            parser.feed(payload.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise AcquisitionError("Directory metadata is not UTF-8", EXIT_NETWORK) from error
        base_parts = urlsplit(base)
        listed: dict[str, str] = {}
        for href in parser.links:
            joined = urljoin(base, href)
            parts = urlsplit(joined)
            if parts.scheme != base_parts.scheme or parts.netloc != base_parts.netloc:
                continue
            if not parts.path.startswith(base_parts.path) or parts.path.endswith("/"):
                continue
            if "/" in parts.path[len(base_parts.path):].strip("/"):
                continue
            safe_url(joined, allow_directory=False)
            listed[self._filename_from_url(joined)] = joined
        member_match = re.search(r"(?:^|;)members=([^;]+)(?:;|$)", row.get("notes", ""))
        if not member_match:
            raise AcquisitionError(f"Directory members are not frozen for {row['resource_id']}", EXIT_PROTOCOL)
        members = member_match.group(1).split(",")
        if not members or any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in members):
            raise AcquisitionError(f"Directory member lock is invalid for {row['resource_id']}", EXIT_PROTOCOL)
        if len(members) != len(set(members)) or not set(members).issubset(listed):
            raise AcquisitionError(f"A locked directory member is unavailable for {row['resource_id']}", EXIT_NETWORK)
        candidates = [listed[name] for name in members]
        physicals = []
        total = 0
        for component, url in enumerate(candidates, start=1):
            metadata = self._remote_metadata(url)
            total += metadata.size
            relative = f"{logical_index:03d}_{row['resource_id']}/{self._filename_from_url(url)}"
            physicals.append(PhysicalFile(
                order_index=f"{logical_index:03d}.{component:03d}",
                resource_id=row["resource_id"], accession=row["accession"], role=row["role"],
                url=url, relative_path=relative, expected_bytes=metadata.size,
                provider_algorithm="NA", provider_checksum="NA", metadata=metadata,
            ))
        expected = as_int(row.get("expected_bytes"))
        if expected is None or total != expected:
            raise AcquisitionError(f"Directory aggregate byte count changed for {row['resource_id']}", EXIT_SIZE)
        return physicals

    def _manifest_physical(self, row: dict[str, str], logical_index: int) -> list[PhysicalFile]:
        manifest_url = safe_url(row["data_url"], allow_directory=False)
        payload, metadata = self._read_metadata_bytes(manifest_url)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AcquisitionError("Provider manifest is not UTF-8", EXIT_NETWORK) from error
        base = manifest_url.rsplit("/", 1)[0] + "/"
        records: list[tuple[int, int, str, str]] = []
        for line in text.splitlines():
            md5_match = re.search(r"(?i)(?<![0-9a-f])([0-9a-f]{32})(?![0-9a-f])", line)
            file_match = re.search(r"([^\s]+\.vcf\.gz(?:\.tbi)?)", line)
            if not md5_match or not file_match:
                continue
            filename = file_match.group(1).lstrip("./")
            chromosome_match = re.search(r"(?:chr|chromosome[_-]?)(\d{1,2})(?:\D|$)", filename, re.I)
            if not chromosome_match:
                continue
            chromosome = int(chromosome_match.group(1))
            if chromosome not in self.config["retention_and_capacity"]["chromosome_order"]:
                continue
            kind = 1 if filename.endswith(".tbi") else 0
            child_url = urljoin(base, filename)
            base_parts, child_parts = urlsplit(base), urlsplit(child_url)
            if child_parts.netloc != base_parts.netloc or not child_parts.path.startswith(base_parts.path):
                raise AcquisitionError("Provider manifest contains an unlisted URL", EXIT_PATH)
            safe_url(child_url, allow_directory=False)
            records.append((chromosome, kind, child_url, md5_match.group(1).lower()))
        chromosomes = self.config["retention_and_capacity"]["chromosome_order"]
        if len(records) != 44 or {(chromosome, kind) for chromosome, kind, _, _ in records} != {
            (chromosome, kind) for chromosome in chromosomes for kind in (0, 1)
        }:
            raise AcquisitionError("The chromosome provider manifest does not contain the locked 44 files", EXIT_ORDER)
        records.sort(key=lambda item: (item[0], item[1]))

        manifest_relative = f"{logical_index:03d}_{row['resource_id']}/provider_manifest/{self._filename_from_url(manifest_url)}"
        physicals = [PhysicalFile(
            order_index=f"{logical_index:03d}.000",
            resource_id=row["resource_id"], accession=row["accession"], role=row["role"],
            url=manifest_url, relative_path=manifest_relative, expected_bytes=len(payload),
            provider_algorithm="NA", provider_checksum="NA", metadata=metadata, prefetched_bytes=payload,
        )]
        aggregate = 0
        for component, (_, _, url, checksum) in enumerate(records, start=1):
            child_metadata = self._remote_metadata(url)
            aggregate += child_metadata.size
            relative = f"{logical_index:03d}_{row['resource_id']}/{self._filename_from_url(url)}"
            physicals.append(PhysicalFile(
                order_index=f"{logical_index:03d}.{component:03d}",
                resource_id=row["resource_id"], accession=row["accession"], role=row["role"],
                url=url, relative_path=relative, expected_bytes=child_metadata.size,
                provider_algorithm="MD5", provider_checksum=checksum, metadata=child_metadata,
            ))
        expected = as_int(row.get("expected_bytes"))
        if expected is None or aggregate != expected:
            raise AcquisitionError(f"Chromosome aggregate byte count changed for {row['resource_id']}", EXIT_SIZE)
        return physicals

    def _expand(self, row: dict[str, str], logical_index: int) -> list[PhysicalFile]:
        if row.get("provider_checksum_algorithm") == "manifest_MD5_set":
            return self._manifest_physical(row, logical_index)
        if row["data_url"].endswith("/"):
            return self._directory_physical(row, logical_index)
        return self._direct_physical(row, logical_index)

    @staticmethod
    def _filename_from_url(url: str) -> str:
        filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise AcquisitionError("Remote filename is unsafe", EXIT_PATH)
        return filename

    @staticmethod
    def _verify_etag_lock(physical: PhysicalFile) -> None:
        if physical.provider_algorithm != "ETag":
            return
        expected = physical.provider_checksum.removeprefix("HTTP_ETag_").strip('"')
        observed = physical.metadata.etag.removeprefix("W/").strip('"')
        if not observed or observed != expected:
            raise AcquisitionError(f"Provider ETag changed for {physical.resource_id}", EXIT_CHECKSUM)

    @staticmethod
    def _strong_etag(metadata: RemoteMetadata) -> str:
        etag = metadata.etag.strip()
        if etag.startswith("W/") or len(etag) < 2 or not (etag.startswith('"') and etag.endswith('"')):
            return ""
        return etag

    @classmethod
    def _validator(cls, metadata: RemoteMetadata) -> tuple[str, str]:
        etag = cls._strong_etag(metadata)
        if etag:
            return "etag", etag
        if metadata.last_modified:
            return "last_modified", metadata.last_modified
        return "", ""

    @classmethod
    def _conditional_headers(cls, metadata: RemoteMetadata) -> dict[str, str]:
        kind, value = cls._validator(metadata)
        if kind == "etag":
            return {"If-Match": value}
        if kind == "last_modified":
            return {"If-Unmodified-Since": value}
        return {}

    def _part_state_path(self, part: Path) -> Path:
        return part.with_name(part.name + ".state.json")

    def _write_part_state(
        self,
        part: Path,
        physical: PhysicalFile,
        prefix_bytes: int,
        prefix_sha256: str | None = None,
    ) -> None:
        kind, value = self._validator(physical.metadata)
        if prefix_sha256 is None:
            prefix_sha256 = hash_file(part) if part.is_file() else hashlib.sha256(b"").hexdigest()
        state = {
            "schema_version": 1,
            "url": physical.url,
            "expected_bytes": physical.expected_bytes,
            "protocol_sha256": hash_file(self.config_path),
            "resource_table_sha256": hash_file(self.resources_path),
            "validator_kind": kind,
            "validator": value,
            "prefix_bytes": prefix_bytes,
            "prefix_sha256": prefix_sha256,
            "updated_utc": utc_now(),
        }
        payload = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8")
        atomic_bytes(self._part_state_path(part), payload)

    def _valid_part_state(
        self,
        part: Path,
        physical: PhysicalFile,
        current: RemoteMetadata | None = None,
    ) -> bool:
        state_path = self._part_state_path(part)
        if not state_path.is_file() or _is_reparse(state_path):
            return False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        kind, value = self._validator(physical.metadata)
        expected = {
            "schema_version": 1,
            "url": physical.url,
            "expected_bytes": physical.expected_bytes,
            "protocol_sha256": hash_file(self.config_path),
            "resource_table_sha256": hash_file(self.resources_path),
            "validator_kind": kind,
            "validator": value,
        }
        if not value or any(state.get(key) != expected_value for key, expected_value in expected.items()):
            return False
        if current is not None and self._validator(current) != (kind, value):
            return False
        if state.get("prefix_bytes") != part.stat().st_size or state.get("prefix_bytes", -1) > physical.expected_bytes:
            return False
        recorded_sha256 = state.get("prefix_sha256", "")
        return bool(re.fullmatch(r"[0-9a-f]{64}", recorded_sha256)) and hash_file(part) == recorded_sha256

    def _remove_part_state(self, part: Path) -> None:
        state = self._part_state_path(part)
        if state.exists():
            if _is_reparse(state) or not state.is_file():
                raise AcquisitionError("Partial state path is unsafe", EXIT_PATH)
            state.unlink()

    def _current_metadata(self, physical: PhysicalFile) -> RemoteMetadata:
        current = self._remote_metadata(physical.url)
        if current.size != physical.expected_bytes:
            raise AcquisitionError(f"Remote byte count changed for {physical.resource_id}", EXIT_SIZE)
        locked_validator = self._validator(physical.metadata)
        current_validator = self._validator(current)
        if locked_validator != current_validator:
            raise AcquisitionError(f"Remote validator changed for {physical.resource_id}", EXIT_CHECKSUM)
        return current

    def _verify_provider(self, path: Path, physical: PhysicalFile) -> tuple[str, str]:
        algorithm = physical.provider_algorithm
        expected = physical.provider_checksum
        if algorithm in NA_VALUES or expected in NA_VALUES:
            return "unavailable", ""
        if algorithm == "MD5":
            observed = hash_file(path, "md5")
        elif algorithm == "SHA256":
            observed = hash_file(path, "sha256")
        elif algorithm == "Git_blob_SHA1":
            observed = git_blob_sha1(path)
        elif algorithm == "ETag":
            self._verify_etag_lock(physical)
            return "verified", physical.metadata.etag
        else:
            raise AcquisitionError(f"Unsupported provider checksum algorithm: {algorithm}", EXIT_PROTOCOL)
        if observed.lower() != expected.lower():
            raise AcquisitionError(f"Provider checksum mismatch for {physical.resource_id}", EXIT_CHECKSUM)
        return "verified", observed.lower()

    def _transfer(self, physical: PhysicalFile) -> tuple[int, str, str, RemoteMetadata]:
        destination = safe_destination(self.data_root, physical.relative_path)
        ensure_safe_directory(destination.parent)
        if destination.exists():
            if not destination.is_file() or _is_reparse(destination):
                raise AcquisitionError("Downloaded destination is not a regular file", EXIT_PATH)
            if destination.stat().st_size != physical.expected_bytes:
                raise AcquisitionError(f"Existing source byte count mismatch for {physical.resource_id}", EXIT_SIZE)
            provider_status, _ = self._verify_provider(destination, physical)
            sha256 = hash_file(destination)
            prior = next((entry for entry in self.manifest["entries"] if entry.get("url") == physical.url), None)
            if prior and prior.get("sha256") != sha256:
                raise AcquisitionError(f"Existing source SHA256 changed for {physical.resource_id}", EXIT_CHECKSUM)
            return 0, provider_status, sha256, physical.metadata

        part = destination.with_name(destination.name + ".part")
        if part.exists() and (_is_reparse(part) or not part.is_file()):
            raise AcquisitionError("Partial destination is not a regular file", EXIT_PATH)
        if part.is_file() and part.stat().st_size == physical.expected_bytes:
            if self._valid_part_state(part, physical):
                provider_status, _ = self._verify_provider(part, physical)
                sha256 = hash_file(part)
                os.replace(part, destination)
                self._remove_part_state(part)
                return 0, provider_status, sha256, physical.metadata
            part.unlink()
            self._remove_part_state(part)
        current = self._current_metadata(physical)
        active = replace(physical, metadata=current)
        retry_count = 0
        max_redownloads = int(self.config["checksums"]["redownload_attempts"])
        while True:
            try:
                retry_count += self._write_part(part, active, retry_count < max_redownloads)
                if part.stat().st_size != active.expected_bytes:
                    raise AcquisitionError(f"Downloaded byte count mismatch for {physical.resource_id}", EXIT_SIZE)
                provider_status, _ = self._verify_provider(part, active)
                sha256 = hash_file(part)
                os.replace(part, destination)
                self._remove_part_state(part)
                return retry_count, provider_status, sha256, current
            except AcquisitionError as error:
                if error.exit_code not in {EXIT_NETWORK, EXIT_SIZE, EXIT_CHECKSUM} or retry_count >= max_redownloads:
                    error.retry_count = retry_count
                    raise
                if part.exists() and error.exit_code in {EXIT_SIZE, EXIT_CHECKSUM}:
                    part.unlink()
                    self._remove_part_state(part)
                retry_count += 1

    def _write_part(self, part: Path, physical: PhysicalFile, allow_restart: bool) -> int:
        if physical.prefetched_bytes is not None:
            self._write_bytes_to_part(part, physical.prefetched_bytes, "wb")
            self._write_part_state(
                part,
                physical,
                len(physical.prefetched_bytes),
                hashlib.sha256(physical.prefetched_bytes).hexdigest(),
            )
            return 0

        offset = part.stat().st_size if part.exists() else 0
        if offset > physical.expected_bytes:
            part.unlink()
            self._remove_part_state(part)
            offset = 0
        valid_state = offset and self._valid_part_state(part, physical, physical.metadata)
        if offset and not valid_state:
            part.unlink()
            self._remove_part_state(part)
            offset = 0
        if offset and physical.metadata.accepts_ranges and valid_state:
            headers = {"Range": f"bytes={offset}-", **self._conditional_headers(physical.metadata)}
            response = self._request(physical.url, headers=headers)
            expected_range = f"bytes {offset}-{physical.expected_bytes - 1}/{physical.expected_bytes}"
            status = getattr(response, "status", response.getcode())
            if (
                status == 206
                and response.headers.get("Content-Range", "") == expected_range
                and self._same_remote_entity(response, physical.metadata)
            ):
                self._stream_response(response, part, "ab", physical.expected_bytes - offset, physical)
                return 0
            response.close()
            if not allow_restart:
                raise AcquisitionError("Invalid range response exhausted the redownload allowance", EXIT_NETWORK)
            self._stream_full(physical, part)
            return 1
        self._stream_full(physical, part)
        return 0

    def _stream_full(self, physical: PhysicalFile, part: Path) -> None:
        response = self._request(physical.url, headers=self._conditional_headers(physical.metadata))
        status = getattr(response, "status", response.getcode())
        length = response.headers.get("Content-Length", "")
        checksum_locks_identity = (
            physical.provider_algorithm in {"MD5", "SHA256", "Git_blob_SHA1"}
            and physical.provider_checksum not in NA_VALUES
        )
        if (
            status != 200
            or length != str(physical.expected_bytes)
            or not self._same_remote_entity(
                response,
                physical.metadata,
                allow_missing=checksum_locks_identity,
            )
        ):
            response.close()
            raise AcquisitionError("Full transfer response does not match locked metadata", EXIT_NETWORK)
        self._stream_response(response, part, "wb", physical.expected_bytes, physical)

    @staticmethod
    def _same_remote_entity(response, metadata: RemoteMetadata, allow_missing: bool = False) -> bool:
        observed_etag = response.headers.get("ETag", "").strip()
        observed_modified = response.headers.get("Last-Modified", "")
        if metadata.etag and observed_etag and observed_etag != metadata.etag:
            return False
        if metadata.last_modified and observed_modified and observed_modified != metadata.last_modified:
            return False
        if not allow_missing and metadata.etag and not observed_etag:
            return False
        if not allow_missing and metadata.last_modified and not observed_modified:
            return False
        return True

    @staticmethod
    def _write_bytes_to_part(path: Path, payload: bytes, mode: str) -> None:
        try:
            with GeneticInputAcquirer._open_part(path, mode) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except AcquisitionError:
            raise
        except OSError as error:
            if error.errno == errno.ENOSPC:
                raise AcquisitionError("Free space was exhausted during transfer", EXIT_SPACE) from error
            raise AcquisitionError(f"Body write failed: {error}", EXIT_NETWORK) from error

    @staticmethod
    def _open_part(path: Path, mode: str):
        if path.exists() and (_is_reparse(path) or not path.is_file()):
            raise AcquisitionError("Partial path is unsafe", EXIT_PATH)
        flags = os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if mode == "ab":
            flags |= os.O_APPEND
        elif path.exists():
            flags |= os.O_TRUNC
        else:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise AcquisitionError(f"Cannot open partial file safely: {error}", EXIT_PATH) from error
        return os.fdopen(descriptor, mode)

    def _stream_response(
        self,
        response,
        path: Path,
        mode: str,
        expected_bytes: int,
        physical: PhysicalFile,
    ) -> None:
        observed = 0
        prefix = path.stat().st_size if mode == "ab" and path.exists() else 0
        digest = hashlib.sha256()
        if prefix:
            with path.open("rb") as existing:
                for chunk in iter(lambda: existing.read(CHUNK_BYTES), b""):
                    digest.update(chunk)
        try:
            with response, self._open_part(path, mode) as handle:
                reader = getattr(response, "read1", response.read)
                for chunk in iter(lambda: reader(CHUNK_BYTES), b""):
                    handle.write(chunk)
                    observed += len(chunk)
                    digest.update(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                    self._write_part_state(path, physical, prefix + observed, digest.hexdigest())
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            if error.errno == errno.ENOSPC:
                raise AcquisitionError("Free space was exhausted during transfer", EXIT_SPACE) from error
            raise AcquisitionError(f"Body read or write failed: {error}", EXIT_NETWORK) from error
        except (HTTPError, URLError, TimeoutError, http.client.HTTPException) as error:
            raise AcquisitionError(f"Body read failed: {error}", EXIT_NETWORK) from error
        if observed != expected_bytes:
            raise AcquisitionError("Transfer body ended before the locked byte count", EXIT_NETWORK)

    def _entry_for(self, physical: PhysicalFile, retries: int, provider_status: str, sha256: str) -> dict:
        prior = next((entry for entry in self.manifest["entries"] if entry.get("url") == physical.url), None)
        completed = (prior or {}).get("completed_utc") or utc_now()
        started = (prior or {}).get("started_utc") or physical.metadata.checked_utc
        retry_count = (prior or {}).get("retry_count", retries) if prior else retries
        return {
            "order_index": physical.order_index,
            "resource_id": physical.resource_id,
            "accession": physical.accession,
            "role": physical.role,
            "url": physical.url,
            "relative_path": physical.relative_path,
            "expected_bytes": physical.expected_bytes,
            "bytes": physical.expected_bytes,
            "provider_checksum_algorithm": physical.provider_algorithm,
            "provider_checksum": physical.provider_checksum,
            "provider_checksum_status": provider_status,
            "sha256": sha256,
            "metadata_checked_utc": physical.metadata.checked_utc,
            "started_utc": started,
            "completed_utc": completed,
            "retry_count": retry_count,
            "completion_state": "complete",
        }

    def _missing_entry(self, physical: PhysicalFile) -> dict:
        return {
            "order_index": physical.order_index,
            "resource_id": physical.resource_id,
            "accession": physical.accession,
            "role": physical.role,
            "url": physical.url,
            "relative_path": physical.relative_path,
            "expected_bytes": physical.expected_bytes,
            "bytes": 0,
            "provider_checksum_algorithm": physical.provider_algorithm,
            "provider_checksum": physical.provider_checksum,
            "provider_checksum_status": "not_verified",
            "sha256": "",
            "metadata_checked_utc": physical.metadata.checked_utc,
            "started_utc": "",
            "completed_utc": "",
            "retry_count": 0,
            "completion_state": "missing",
        }

    def _store_entry(self, entry: dict) -> None:
        entries = [existing for existing in self.manifest["entries"] if existing.get("url") != entry["url"]]
        entries.append(entry)
        entries.sort(key=lambda item: tuple(int(part) for part in item["order_index"].split(".")))
        self.manifest["entries"] = entries

    def _store_failed_entry(self, physical: PhysicalFile, error: AcquisitionError) -> None:
        destination = safe_destination(self.data_root, physical.relative_path)
        part = destination.with_name(destination.name + ".part")
        observed = part.stat().st_size if part.is_file() and not part.is_symlink() else 0
        entry = {
            "order_index": physical.order_index,
            "resource_id": physical.resource_id,
            "accession": physical.accession,
            "role": physical.role,
            "url": physical.url,
            "relative_path": physical.relative_path,
            "expected_bytes": physical.expected_bytes,
            "bytes": observed,
            "provider_checksum_algorithm": physical.provider_algorithm,
            "provider_checksum": physical.provider_checksum,
            "provider_checksum_status": "failed" if error.exit_code == EXIT_CHECKSUM else "not_verified",
            "sha256": "",
            "metadata_checked_utc": physical.metadata.checked_utc,
            "started_utc": physical.metadata.checked_utc,
            "completed_utc": "",
            "retry_count": getattr(error, "retry_count", 0),
            "completion_state": "stopped",
            "failure_exit_code": error.exit_code,
        }
        self._store_entry(entry)

    def _cleanup_parts(self) -> int:
        if not self.data_root.exists():
            return 0
        removed = 0
        for part in self.data_root.rglob("*.part"):
            resolved = Path(os.path.abspath(part))
            try:
                resolved.relative_to(self.data_root)
            except ValueError as error:
                raise AcquisitionError("Partial file escapes the locked data root", EXIT_PATH) from error
            if _is_reparse(part) or not part.is_file():
                raise AcquisitionError("Partial cleanup target is not a regular file", EXIT_PATH)
            part.unlink()
            self._remove_part_state(part)
            removed += 1
        return removed

    def _reconcile(self, physicals: list[PhysicalFile], fill_missing: bool) -> bool:
        roster_urls = {physical.url for physical in physicals}
        if any(entry.get("url") not in roster_urls for entry in self.manifest["entries"]):
            raise AcquisitionError("Checksum manifest URL is absent from the physical roster", EXIT_ORDER)
        complete_prefix = True
        rebuilt = []
        self.verified_urls.clear()
        prior_by_url = {entry.get("url"): entry for entry in self.manifest["entries"]}
        for physical in physicals:
            destination = safe_destination(self.data_root, physical.relative_path)
            if destination.exists():
                if not complete_prefix:
                    raise AcquisitionError("A completed file appears after the first incomplete roster item", EXIT_ORDER)
                if not destination.is_file() or _is_reparse(destination):
                    raise AcquisitionError("Roster destination is not a regular file", EXIT_PATH)
                if destination.stat().st_size != physical.expected_bytes:
                    raise AcquisitionError("Roster destination byte count changed", EXIT_SIZE)
                provider_status, _ = self._verify_provider(destination, physical)
                sha256 = hash_file(destination)
                prior = prior_by_url.get(physical.url)
                if prior and prior.get("completion_state") == "complete" and prior.get("sha256") != sha256:
                    raise AcquisitionError("Roster destination SHA256 changed", EXIT_CHECKSUM)
                entry = self._entry_for(physical, 0, provider_status, sha256)
                rebuilt.append(entry)
                self.verified_urls.add(physical.url)
            else:
                complete_prefix = False
                prior = prior_by_url.get(physical.url)
                if prior and prior.get("completion_state") == "complete":
                    raise AcquisitionError("A manifest-complete source file is missing", EXIT_CHECKSUM)
                if fill_missing:
                    rebuilt.append(self._missing_entry(physical))
                elif prior:
                    rebuilt.append(prior)
        if fill_missing:
            self.manifest["entries"] = rebuilt
        else:
            for entry in rebuilt:
                self._store_entry(entry)
        return len(self.verified_urls) == len(physicals)

    def _remaining_bytes(self, physicals: list[PhysicalFile]) -> int:
        remaining = 0
        for physical in physicals:
            if physical.url in self.verified_urls:
                continue
            destination = safe_destination(self.data_root, physical.relative_path)
            part = destination.with_name(destination.name + ".part")
            reserved = 0
            if part.exists():
                if _is_reparse(part) or not part.is_file():
                    raise AcquisitionError("Partial path is unsafe", EXIT_PATH)
                reserved = min(part.stat().st_size, physical.expected_bytes)
            remaining += physical.expected_bytes - reserved
        return remaining

    def manifest_only(self, physicals: list[PhysicalFile] | None) -> int:
        if physicals is None:
            error = AcquisitionError("A complete local acquisition roster is required for manifest-only verification", EXIT_INCOMPLETE)
            self._write_manifest("incomplete", error)
            return EXIT_INCOMPLETE
        complete = self._reconcile(physicals, fill_missing=True)
        state = "complete" if complete else "incomplete"
        self._write_manifest(state)
        return EXIT_OK if state == "complete" else EXIT_INCOMPLETE

    def run(
        self,
        dry_run: bool = False,
        max_files: int | None = None,
        manifest_only: bool = False,
        remove_stale_parts: bool = False,
    ) -> int:
        self.load_and_validate()
        if max_files is not None and max_files < 1:
            raise AcquisitionError("--max-files must be at least 1", EXIT_USAGE)
        with DataRootLock(self.data_root):
            self.manifest = self._load_manifest()
            if remove_stale_parts:
                removed = self._cleanup_parts()
                self.logger.info("Removed %d explicitly selected partial files", removed)
            try:
                physicals = self._load_roster()
                if manifest_only:
                    return self.manifest_only(physicals)
                if physicals is None:
                    initial_remaining = sum(int(row["budget_bytes"]) for row in self.plan)
                    self._disk_gate(initial_remaining)
                    physicals = self._build_roster()
                self._reconcile(physicals, fill_missing=False)
                self._disk_gate(self._remaining_bytes(physicals))
            except AcquisitionError as error:
                if not dry_run:
                    self._write_manifest("stopped", error)
                raise

            processed = 0
            try:
                for physical in physicals:
                    if max_files is not None and processed >= max_files:
                        if not dry_run:
                            self._write_manifest("partial")
                        return EXIT_OK
                    self._disk_gate(self._remaining_bytes(physicals))
                    if dry_run:
                        self.logger.info(
                            "Planned %s role=%s bytes=%d",
                            physical.resource_id,
                            physical.role,
                            physical.expected_bytes,
                        )
                    elif physical.url not in self.verified_urls:
                        try:
                            retries, provider_status, sha256, current = self._transfer(physical)
                        except AcquisitionError as error:
                            self._store_failed_entry(physical, error)
                            raise
                        completed_physical = replace(physical, metadata=current)
                        self._store_entry(self._entry_for(completed_physical, retries, provider_status, sha256))
                        self.verified_urls.add(physical.url)
                        self._write_manifest("partial")
                        self.logger.info(
                            "Completed %s role=%s bytes=%d",
                            physical.resource_id,
                            physical.role,
                            physical.expected_bytes,
                        )
                    processed += 1
                if not dry_run:
                    self._write_manifest("complete")
                return EXIT_OK
            except AcquisitionError as error:
                if not dry_run:
                    self._write_manifest("stopped", error)
                raise


def default_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "config" / "genetic_confirmation_protocol.yaml",
        root / "analyses" / "independent_confirmation" / "genetic_protocol_resources.tsv",
        root / "data" / "independent_confirmation" / "genetics",
    )


def build_parser() -> argparse.ArgumentParser:
    config, resources, data_root = default_paths()
    parser = argparse.ArgumentParser(description="Acquire locked genetic confirmation inputs as opaque files.")
    parser.add_argument("--config", type=Path, default=config)
    parser.add_argument("--resources", type=Path, default=resources)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--remove-stale-parts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        acquirer = GeneticInputAcquirer(
            config_path=args.config,
            resources_path=args.resources,
            data_root=args.data_root,
            manifest_path=args.manifest,
            timeout=args.timeout,
        )
        return acquirer.run(
            dry_run=args.dry_run,
            max_files=args.max_files,
            manifest_only=args.manifest_only,
            remove_stale_parts=args.remove_stale_parts,
        )
    except AcquisitionError as error:
        logging.error("%s", error)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
