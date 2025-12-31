"""Freeze inputs and rules for the regulatory verification extension."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "config/molecular_cascade.yaml",
    "docs/protocol.md",
    "data/interim/gwt_vectors/rows.parquet",
    "data/interim/gwt_vectors/genes.parquet",
    "data/interim/gwt_vectors/normalized_significant_logfc.npz",
    "analyses/primary/results/transfer_phenotypes.parquet",
    "analyses/primary/results/context_switching.parquet",
    "analyses/vectors/results/contextual_transfer_tensor.parquet",
    "analyses/vectors/results/context_rerouting_pairs.parquet",
    "analyses/vectors/results/network_gain_rows.parquet",
    "analyses/replication/results/k562_replication_rows.parquet",
    "analyses/natural_genetics/results/directional_pairs.parquet",
    "analyses/disease/results/disease_model_rows.parquet",
    "analyses/external_benchmark/results/all_eligible_target_predictions.parquet",
    "analyses/guide_dose_response/results/eligible_guide_rows.csv",
    "analyses/guide_dose_response/results/hypothesis_decisions.csv",
    "analyses/causal_triangulation/results/evidence_gates.csv",
    "work/upstream/GWT_perturbseq_analysis_2025/metadata/DE_by_guide.correlation_results.csv",
    "work/upstream/GWT_perturbseq_analysis_2025/metadata/suppl_tables/guide_kd_efficiency.suppl_table.csv",
    "work/motif/TRANSFAC_and_JASPAR_PWMs.gmt",
    "work/cd4_pchic_interactions.tsv.gz",
    "work/causal_inputs/ENCFF944LFH.bed.gz",
    "work/causal_inputs/ENCFF068XUG.bed.gz",
    "work/regulatory_inputs/ENCFF858TLX.bed.gz",
    "work/regulatory_inputs/omnipath_tf_target.tsv",
    "work/regulatory_inputs/ensembl_gene_attributes.tsv",
    "work/regulatory_inputs/hg19ToHg38.over.chain.gz",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing artifacts: {missing}")
    config = yaml.safe_load((ROOT / "config/molecular_cascade.yaml").read_text(encoding="utf-8"))
    rows = pd.read_parquet(ROOT / "data/interim/gwt_vectors/rows.parquet", columns=["target_contrast_gene_name", "culture_condition"])
    genes = pd.read_parquet(ROOT / "data/interim/gwt_vectors/genes.parquet", columns=["gene_name"])
    if sorted(rows["culture_condition"].unique()) != sorted(config["design"]["primary_states"]):
        raise ValueError("State denominator disagrees with frozen configuration")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    manifest = {
        "freeze_date": "2026-09-13",
        "status": "retrospective_rules_frozen_before_regulatory_join",
        "not_a_prospective_preregistration": True,
        "primary_target_state_rows": int(len(rows)),
        "primary_targets": int(rows["target_contrast_gene_name"].nunique()),
        "measured_response_genes": int(genes["gene_name"].nunique()),
        "states": sorted(rows["culture_condition"].unique()),
        "git_commit_before_freeze_commit": commit,
        "artifacts": [
            {
                "path": name.replace("\\", "/"),
                "bytes": (ROOT / name).stat().st_size,
                "sha256": sha256(ROOT / name),
            }
            for name in FILES
        ],
    }
    output = ROOT / "analyses/molecular_cascade/freeze_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
