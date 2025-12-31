"""Run the frozen genetic-confirmation stages in locked order."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/independent_confirmation/genetics"
OUTPUT = ROOT / "analyses/independent_confirmation/results"
MANIFEST = DATA / "checksum_manifest.json"
ORDER = ["N05", "N02", "G01", "G02", "G03", "G04", "G05"]
EXPECTED_DENSE_VARIANTS = {resource_id: 44_300_000 for resource_id in ["N05", "N02", "G01", "G03", "G04", "G05"]}
LOCKED_STUDY_N = {"N05": 449196, "N02": 440382, "G01": 315668, "G02": 1420658, "G03": 450483, "G04": 435738, "G05": 447331}
REQUIRED_FREE_BYTES = 91_510_576_128
AUTOSOMES = {str(value) for value in range(1, 23)}
ALLELES = {"A", "C", "G", "T"}
PROGRAM_MANIFEST_HASH = "54579caf9c69001aa2f70edf352f59b8eb1235f537231a698c580afbc0224ef8"
PROGRAM_FILES = [
    ("analyses/context_dynamics/results/switch_scores.parquet", 250772, "8431f325c278e9cedf6c456ad917aa12fdddc0b492d03290f5a274fa28877a7a"),
    ("analyses/molecular_cascade/results/edge_evidence_matrix.parquet", 1867568, "463185e912e5370b5fdd49afd44789f3ff87cc91000e2cfb3730835f0f337785"),
    ("analyses/primary/results/transfer_phenotypes.parquet", 1991787, "4327ea47df56278e92a60c8e4a26c41f5ff2d8f4e4786108e23e8998d0f97b3f"),
    ("analyses/state_response_decomposition/results/target_state_decomposition.parquet", 713649, "576e9d12e22aa4f790d4f4f535de3f40480911857ee8959c705a5a51a2174863"),
    ("analyses/vectors/results/reference_module_loadings.parquet", 3040851, "85a083525b34f2fa321eb43f7d71a42270ee5469306b4cf16cc56403b34739ef"),
]
GZIP_OPTIONS = {"method": "gzip", "mtime": 0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def stage_lock(resource_id: str):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f".{resource_id}.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"stage {resource_id} is already running") from error
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_acquisition() -> tuple[dict[str, object], str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sidecar = (MANIFEST.with_suffix(MANIFEST.suffix + ".sha256")).read_text(encoding="utf-8").split()[0]
    manifest_hash = sha256(MANIFEST)
    if manifest_hash != sidecar or manifest.get("completion_state") != "complete":
        raise RuntimeError("acquisition manifest is not verified complete")
    entries = manifest.get("entries", [])
    if len(entries) != 82 or any(entry.get("completion_state") != "complete" for entry in entries):
        raise RuntimeError("physical acquisition denominator is incomplete")
    residue = list(DATA.rglob("*.part")) + list(DATA.rglob("*.part.state.json")) + list(DATA.rglob("*.txn")) + list(DATA.rglob("*.tmp"))
    if residue:
        raise RuntimeError("unresolved acquisition transaction files remain")
    if shutil.disk_usage(ROOT).free < REQUIRED_FREE_BYTES:
        raise RuntimeError("frozen free-space gate failed")
    program_rows: list[tuple[str, str]] = []
    for relative_path, expected_bytes, expected_sha256 in PROGRAM_FILES:
        source = ROOT / relative_path
        if source.stat().st_size != expected_bytes or sha256(source) != expected_sha256:
            raise RuntimeError(f"frozen program file changed: {relative_path}")
        program_rows.append((relative_path, expected_sha256))
    frozen_text = "".join(f"{path}\t{digest}\n" for path, digest in sorted(program_rows))
    program_hash = hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()
    if program_hash != PROGRAM_MANIFEST_HASH or program_hash != manifest["program_manifest_sha256"]:
        raise RuntimeError("frozen program manifest changed")
    return manifest, manifest_hash


def resource_entry(manifest: dict[str, object], resource_id: str) -> dict[str, object]:
    matches = [entry for entry in manifest["entries"] if entry["resource_id"] == resource_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one physical GWAS file for {resource_id}")
    entry = matches[0]
    path = DATA / entry["relative_path"]
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise RuntimeError(f"source identity changed for {resource_id}")
    return entry


def enforce_opening_order(resource_id: str) -> None:
    if resource_id not in ORDER:
        raise RuntimeError(f"resource {resource_id} is not a locked disease outcome")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current_hashes = {entry["resource_id"]: entry["sha256"] for entry in manifest["entries"] if entry["resource_id"] in ORDER}
    for prior in ORDER[: ORDER.index(resource_id)]:
        path = OUTPUT / f"schema_{prior}.json"
        if not path.exists():
            raise RuntimeError(f"locked predecessor {prior} is incomplete")
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("resource_id") != prior or audit.get("status") not in {"passed", "failed"} or audit.get("source_sha256") != current_hashes.get(prior):
            raise RuntimeError(f"locked predecessor {prior} audit is invalid")
    if resource_id.startswith("G"):
        gate_path = OUTPUT / "negative_pipeline_complete.json"
        if not gate_path.exists():
            raise RuntimeError("positive outcomes remain locked until both negative pipelines are complete")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "complete" or gate.get("resources") != ["N05", "N02"] or gate.get("manifest_sha256") != sha256(MANIFEST):
            raise RuntimeError("negative pipeline completion record is invalid")


def numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


def scan_gwas(resource_id: str, chunk_size: int) -> None:
    with stage_lock(resource_id):
        try:
            _scan_gwas(resource_id, chunk_size)
        except (pd.errors.ParserError, UnicodeDecodeError, EOFError, gzip.BadGzipFile) as error:
            for path in [OUTPUT / "harmonized" / f"{resource_id}.partial.tsv.gz", OUTPUT / f"significant_{resource_id}.partial.tsv.gz"]:
                path.unlink(missing_ok=True)
            manifest, manifest_hash = verify_acquisition()
            entry = resource_entry(manifest, resource_id)
            atomic_json(
                OUTPUT / f"schema_{resource_id}.json",
                {
                    "error": f"{type(error).__name__}: {error}",
                    "manifest_sha256": manifest_hash,
                    "outputs": [],
                    "resource_id": resource_id,
                    "role": entry["role"],
                    "source_bytes": entry["bytes"],
                    "source_sha256": entry["sha256"],
                    "status": "failed",
                    "stop_reasons": ["truncated_or_malformed_file"],
                },
            )


def _scan_gwas(resource_id: str, chunk_size: int) -> None:
    manifest, manifest_hash = verify_acquisition()
    enforce_opening_order(resource_id)
    entry = resource_entry(manifest, resource_id)
    audit_path = OUTPUT / f"schema_{resource_id}.json"
    if audit_path.exists():
        prior = json.loads(audit_path.read_text(encoding="utf-8"))
        if prior.get("source_sha256") != entry["sha256"]:
            raise RuntimeError(f"existing audit source changed for {resource_id}")
        if prior.get("status") not in {"passed", "failed"}:
            raise RuntimeError(f"existing audit status is invalid for {resource_id}")
        if prior.get("status") == "passed":
            for output in prior.get("outputs", []):
                output_path = ROOT / output["path"]
                if not output_path.is_file() or sha256(output_path) != output["sha256"]:
                    raise RuntimeError(f"existing output changed for {resource_id}: {output['path']}")
        print(json.dumps({"resource_id": resource_id, "status": prior.get("status"), "action": "already_complete"}))
        return
    path = DATA / entry["relative_path"]
    compression = "gzip" if path.suffix == ".gz" else None
    header = pd.read_csv(path, sep="\t", compression=compression, nrows=0).columns.tolist()
    fixed = [
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "standard_error",
        "p_value",
    ]
    sample_size_field = "n" if "n" in header else None
    effect = [name for name in ["beta", "odds_ratio"] if name in header]
    frequency_fields = [name for name in ["effect_allele_frequency", "minor_allele_frequency"] if name in header]
    frequency = frequency_fields[0] if len(frequency_fields) == 1 else None
    info = "r2" if "r2" in header else "info" if "info" in header else None
    variant = "rsid" if "rsid" in header else "variant_id" if "variant_id" in header else None
    missing = [name for name in fixed if name not in header]
    schema_errors: list[str] = []
    if missing:
        schema_errors.append("missing_required_fields:" + ",".join(missing))
    if len(effect) != 1:
        schema_errors.append("effect_field_not_exactly_one")
    if frequency is None:
        schema_errors.append("frequency_field_not_exactly_one")
    if info is None:
        schema_errors.append("imputation_quality_field_missing")
    if schema_errors:
        atomic_json(
            audit_path,
            {
                "resource_id": resource_id,
                "source_sha256": entry["sha256"],
                "status": "failed",
                "stop_reasons": schema_errors,
            },
        )
        return

    columns = fixed + effect + [frequency, info] + ([sample_size_field] if sample_size_field else []) + ([variant] if variant else [])
    counts = {
        "rows": 0,
        "canonical_coordinate_fail": 0,
        "allele_fail": 0,
        "effect_fail": 0,
        "standard_error_fail": 0,
        "p_value_fail": 0,
        "sample_size_fail": 0,
        "frequency_fail": 0,
        "info_range_fail": 0,
        "info_below_0_8": 0,
        "palindrome_high_maf": 0,
        "duplicate_keys": 0,
        "order_fail": 0,
        "harmonized_rows": 0,
        "ldsc_quality_rows": 0,
        "significant_rows": 0,
    }
    chromosome_counts = {str(value): 0 for value in range(1, 23)}
    previous_coordinate = -1
    tail_coordinate: tuple[str, int] | None = None
    tail_keys: set[tuple[str, int, str, str]] = set()
    started = time.time()
    output_dir = OUTPUT / "harmonized"
    output_dir.mkdir(parents=True, exist_ok=True)
    harmonized_path = output_dir / f"{resource_id}.tsv.gz"
    significant_path = OUTPUT / f"significant_{resource_id}.tsv.gz"
    harmonized_temporary = output_dir / f"{resource_id}.partial.tsv.gz"
    significant_temporary = OUTPUT / f"significant_{resource_id}.partial.tsv.gz"
    harmonized_temporary.unlink(missing_ok=True)
    significant_temporary.unlink(missing_ok=True)
    wrote_harmonized = False
    wrote_significant = False

    reader = pd.read_csv(
        path,
        sep="\t",
        compression=compression,
        usecols=columns,
        chunksize=chunk_size,
        na_values=["#NA", "NA", "NaN", ""],
        keep_default_na=True,
        low_memory=False,
    )
    for chunk_index, frame in enumerate(reader, start=1):
        counts["rows"] += len(frame)
        chromosome = frame["chromosome"].astype("string").str.removeprefix("chr")
        position = numeric(frame, "base_pair_location")
        effect_allele = frame["effect_allele"].astype("string").str.upper()
        other_allele = frame["other_allele"].astype("string").str.upper()
        se = numeric(frame, "standard_error")
        p_value = numeric(frame, "p_value")
        sample_size = numeric(frame, sample_size_field) if sample_size_field else pd.Series(float(LOCKED_STUDY_N[resource_id]), index=frame.index)
        raw_frequency = numeric(frame, frequency)
        quality = numeric(frame, info)
        if effect[0] == "odds_ratio":
            odds_ratio = numeric(frame, "odds_ratio")
            effect_ok = np.isfinite(odds_ratio) & (odds_ratio > 0)
            beta = np.log(odds_ratio.where(effect_ok))
        else:
            beta = numeric(frame, "beta")
            effect_ok = np.isfinite(beta)
        coordinate_ok = chromosome.isin(AUTOSOMES) & np.isfinite(position) & (position > 0) & (position <= np.iinfo(np.int64).max) & (np.floor(position) == position)
        allele_ok = effect_allele.isin(ALLELES) & other_allele.isin(ALLELES) & (effect_allele != other_allele)
        se_ok = np.isfinite(se) & (se > 0)
        p_ok = np.isfinite(p_value) & (p_value > 0) & (p_value <= 1)
        n_ok = np.isfinite(sample_size) & (sample_size > 0)
        if frequency == "minor_allele_frequency":
            maf = raw_frequency
            eaf = pd.Series(np.nan, index=frame.index, dtype=float)
            frequency_ok = np.isfinite(maf) & (maf > 0) & (maf <= 0.5)
        else:
            eaf = raw_frequency
            frequency_ok = np.isfinite(eaf) & (eaf > 0) & (eaf < 1)
            maf = np.minimum(eaf, 1 - eaf)
        info_range_ok = np.isfinite(quality) & (quality >= 0) & (quality <= 1)
        palindrome = ((effect_allele == "A") & (other_allele == "T")) | ((effect_allele == "T") & (other_allele == "A")) | ((effect_allele == "C") & (other_allele == "G")) | ((effect_allele == "G") & (other_allele == "C"))
        palindrome_high_maf = palindrome & (maf > 0.42)
        masks = {
            "canonical_coordinate_fail": ~coordinate_ok,
            "allele_fail": ~allele_ok,
            "effect_fail": ~effect_ok,
            "standard_error_fail": ~se_ok,
            "p_value_fail": ~p_ok,
            "sample_size_fail": ~n_ok,
            "frequency_fail": ~frequency_ok,
            "info_range_fail": ~info_range_ok,
            "info_below_0_8": info_range_ok & (quality < 0.80),
            "palindrome_high_maf": palindrome_high_maf,
        }
        for name, mask in masks.items():
            counts[name] += int(mask.sum())

        numeric_chr = pd.to_numeric(chromosome, errors="coerce")
        coordinate_code = numeric_chr * 1_000_000_000 + position
        finite_code = coordinate_code[np.isfinite(coordinate_code)].to_numpy(dtype=np.int64)
        if len(finite_code):
            if previous_coordinate > finite_code[0]:
                counts["order_fail"] += 1
            counts["order_fail"] += int(np.sum(finite_code[1:] < finite_code[:-1]))
            previous_coordinate = int(finite_code[-1])
        key_frame = pd.DataFrame(
            {
                "chromosome": chromosome,
                "position": position,
                "effect_allele": effect_allele,
                "other_allele": other_allele,
            }
        ).loc[coordinate_ok & allele_ok]
        counts["duplicate_keys"] += int(key_frame.duplicated(keep="first").sum())
        if len(key_frame):
            first_coordinate = (str(key_frame.iloc[0]["chromosome"]), int(key_frame.iloc[0]["position"]))
            if tail_coordinate == first_coordinate:
                prefix = key_frame[
                    (key_frame["chromosome"] == first_coordinate[0])
                    & (key_frame["position"] == first_coordinate[1])
                ]
                prefix_keys = {tuple(row) for row in prefix.itertuples(index=False, name=None)}
                counts["duplicate_keys"] += len(prefix_keys & tail_keys)
            final_coordinate = (str(key_frame.iloc[-1]["chromosome"]), int(key_frame.iloc[-1]["position"]))
            tail = key_frame[
                (key_frame["chromosome"] == final_coordinate[0])
                & (key_frame["position"] == final_coordinate[1])
            ]
            tail_coordinate = final_coordinate
            tail_keys = {tuple(row) for row in tail.itertuples(index=False, name=None)}
        for value, count in chromosome[coordinate_ok].value_counts().items():
            if str(value) in chromosome_counts:
                chromosome_counts[str(value)] += int(count)

        common = coordinate_ok & allele_ok & effect_ok & se_ok & p_ok & n_ok & frequency_ok & info_range_ok & ~palindrome_high_maf
        harmonized = common & (quality >= 0.80)
        ldsc = common & (quality >= 0.90) & (maf >= 0.01)
        counts["harmonized_rows"] += int(harmonized.sum())
        counts["ldsc_quality_rows"] += int(ldsc.sum())
        position_int = position.where(coordinate_ok).astype("Int64")
        canonical_id = chromosome + ":" + position_int.astype("string") + ":" + effect_allele + ":" + other_allele
        snp = frame[variant].astype("string").fillna(canonical_id) if variant else canonical_id
        out = pd.DataFrame(
            {
                "SNP": snp,
                "CHR": chromosome,
                "BP": position_int,
                "A1": effect_allele,
                "A2": other_allele,
                "BETA": beta,
                "SE": se,
                "P": p_value,
                "N": sample_size,
                "EAF": eaf,
                "MAF": maf,
                "INFO": quality,
                "LDSC_QUALITY": ldsc,
            }
        )
        selected = out.loc[harmonized]
        if len(selected):
            selected.to_csv(harmonized_temporary, sep="\t", index=False, compression=GZIP_OPTIONS, mode="at", header=not wrote_harmonized)
            wrote_harmonized = True
        significant = out.loc[harmonized & (p_value < 5e-8)]
        counts["significant_rows"] += len(significant)
        if len(significant):
            significant.to_csv(significant_temporary, sep="\t", index=False, compression=GZIP_OPTIONS, mode="at", header=not wrote_significant)
            wrote_significant = True
        if chunk_index % 10 == 0:
            if shutil.disk_usage(ROOT).free < REQUIRED_FREE_BYTES:
                raise RuntimeError("frozen free-space gate failed during schema scan")
            print(json.dumps({"resource_id": resource_id, "rows": counts["rows"], "elapsed_seconds": round(time.time() - started, 1)}), flush=True)

    unresolved_fraction = counts["effect_fail"] / counts["rows"] if counts["rows"] else 1.0
    duplicate_fraction = counts["duplicate_keys"] / counts["rows"] if counts["rows"] else 1.0
    expected_dense = EXPECTED_DENSE_VARIANTS.get(resource_id)
    dense_coverage = counts["rows"] / expected_dense if expected_dense else None
    missing_chromosomes = [chromosome for chromosome, count in chromosome_counts.items() if count == 0]
    stop_reasons: list[str] = []
    if missing_chromosomes:
        stop_reasons.append("missing_chromosome:" + ",".join(missing_chromosomes))
    if duplicate_fraction > 0.001:
        stop_reasons.append("duplicate_fraction_exceeded")
    if unresolved_fraction > 0.05:
        stop_reasons.append("unresolved_direction_fraction_exceeded")
    if dense_coverage is None:
        stop_reasons.append("dense_coverage_reference_unavailable")
    elif dense_coverage < 0.90:
        stop_reasons.append("dense_coverage_failed")
    if counts["order_fail"]:
        stop_reasons.append("duplicate_gate_unavailable_unsorted_source")
    columns_out = ["SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P", "N", "EAF", "MAF", "INFO", "LDSC_QUALITY"]
    if not wrote_harmonized:
        pd.DataFrame(columns=columns_out).to_csv(harmonized_temporary, sep="\t", index=False, compression=GZIP_OPTIONS)
    if not wrote_significant:
        pd.DataFrame(columns=columns_out).to_csv(significant_temporary, sep="\t", index=False, compression=GZIP_OPTIONS)
    outputs: list[dict[str, object]] = []
    status = "failed" if stop_reasons else "passed"
    if status == "passed":
        os.replace(harmonized_temporary, harmonized_path)
        os.replace(significant_temporary, significant_path)
        for output_path, rows in [(harmonized_path, counts["harmonized_rows"]), (significant_path, counts["significant_rows"])]:
            outputs.append(
                {
                    "bytes": output_path.stat().st_size,
                    "path": output_path.relative_to(ROOT).as_posix(),
                    "rows": rows,
                    "sha256": sha256(output_path),
                }
            )
    else:
        harmonized_temporary.unlink(missing_ok=True)
        significant_temporary.unlink(missing_ok=True)
    result = {
        "chromosome_counts": chromosome_counts,
        "counts": counts,
        "dense_coverage_fraction": dense_coverage,
        "duplicate_fraction": duplicate_fraction,
        "elapsed_seconds": time.time() - started,
        "header": header,
        "manifest_sha256": manifest_hash,
        "missing_chromosomes": missing_chromosomes,
        "outputs": outputs,
        "resource_id": resource_id,
        "role": entry["role"],
        "source_bytes": entry["bytes"],
        "source_sha256": entry["sha256"],
        "status": status,
        "stop_reasons": stop_reasons,
        "unresolved_effect_direction_fraction": unresolved_fraction,
    }
    atomic_json(audit_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    schema = subparsers.add_parser("schema")
    schema.add_argument("--resource", required=True, choices=ORDER)
    schema.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    if args.command == "schema":
        scan_gwas(args.resource, args.chunk_size)


if __name__ == "__main__":
    main()
