"""Record guide-concordance inputs before outcome analysis."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "work/upstream/GWT_perturbseq_analysis_2025/metadata/DE_by_guide.correlation_results.csv",
    "work/upstream/GWT_perturbseq_analysis_2025/metadata/suppl_tables/guide_kd_efficiency.suppl_table.csv",
    "work/upstream/GWT_perturbseq_analysis_2025/metadata/sgrna_df_final.csv",
    "data/interim/gwt_vectors/rows.parquet",
    "data/interim/gwt_vectors/genes.parquet",
    "data/interim/gwt_vectors/normalized_significant_logfc.npz",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/disease/results/disease_model_rows.parquet",
    "analyses/molecular_cascade/results/edge_evidence_matrix.parquet",
    "analyses/molecular_cascade/results/audit.json",
]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = ROOT / "analyses/guide_concordance/freeze_manifest.json"
    if output.exists():
        raise SystemExit("Freeze exists; do not overwrite after outcomes.")
    paths = FILES + [str(p.relative_to(args.source_root)).replace("\\", "/") for p in sorted((args.source_root / "analyses/guide_dose_response/results").glob("*")) if p.is_file()]
    records = [{"path": p, "available": (args.source_root / p).is_file(), "sha256": sha256(args.source_root / p) if (args.source_root / p).is_file() else None} for p in paths]
    config = ROOT / "config/guide_concordance.yaml"
    config_hash = hashlib.sha256(config.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    output.write_text(json.dumps({"design": "retrospective extension", "status": "frozen before new guide-concordance outcome analysis", "config_sha256": config_hash, "inputs": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
